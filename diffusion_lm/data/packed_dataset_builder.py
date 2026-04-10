from __future__ import annotations

import os
import re
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HEADER_MAGIC = b"TBIN"
HEADER_VERSION = 1
HEADER_SIZE = 16
_FLUSH_THRESHOLD = 500_000


def _extract_text(args: tuple[dict, str]) -> str:
    row, column = args
    return (row.get(column) or "").strip()


class PackedDatasetBuilder:

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        block_size: int,
        output_path: str = "data",
        subset: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        num_workers: int = 0,
        batch_size: int = 1024,
        token_dtype: str = "auto",
        use_tbin_header: bool = True,
        **load_kwargs: Any,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.output_path = Path(output_path)
        self.subset = subset
        self.split = split
        self.text_column = text_column
        self.num_workers = num_workers if num_workers > 0 else (os.cpu_count() or 1)
        self.batch_size = batch_size
        self.token_dtype = token_dtype
        self.use_tbin_header = use_tbin_header
        self.load_kwargs = load_kwargs

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.eos_id: int = self._resolve_eos(tokenizer)
        self.packed: np.ndarray | None = None
        self._total_tokens: int = 0
        self._total_sequences: int = 0
 
    #  Public API                                                     
    @property
    def total_tokens(self) -> int:
        if self.packed is not None:
            return self.packed.size
        return self._total_tokens

    @property
    def total_sequences(self) -> int:
        if self.packed is not None:
            return len(self.packed)
        return self._total_sequences

    def build(self) -> PackedDatasetBuilder:
        raw_dataset = self._load_dataset()
        self.packed = None
        self._tokenize_and_save(raw_dataset)
        return self

    def build_from_hf_dataset(self, dataset: Any) -> PackedDatasetBuilder:
        """Build packed tokens from an already loaded Hugging Face dataset object."""
        self.packed = None
        self._tokenize_and_save(dataset)
        return self

    @staticmethod
    def to_dataloader(
        bin_path: str | Path,
        block_size: int,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = True,
        pin_memory: bool = True,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.0,
        val_batch_size: Optional[int] = None,
        split_seed: int = 42,
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        resolved_bin_path = Path(bin_path)
        if resolved_bin_path.is_dir():
            resolved_bin_path = resolved_bin_path / "data.bin"
        if not resolved_bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {resolved_bin_path}")

        offset = 0
        dt = np.dtype(np.uint32)
        with open(resolved_bin_path, "rb") as f:
            magic = f.read(4)
            if magic == b"TBIN":
                _version = struct.unpack("<I", f.read(4))[0]
                stored_block_size = struct.unpack("<I", f.read(4))[0]
                dtype_code = struct.unpack("<I", f.read(4))[0]
                offset = 16
                dt = np.dtype(np.uint16) if dtype_code == 2 else np.dtype(np.uint32)
                if stored_block_size != block_size:
                    raise ValueError(
                        f"Block size mismatch for TBIN file {resolved_bin_path}: "
                        f"file block_size={stored_block_size}, requested={block_size}. "
                        "Pass the same block_size used while creating the binary."
                    )

        file_size = resolved_bin_path.stat().st_size
        if file_size < offset:
            raise ValueError(f"Invalid binary file {resolved_bin_path}: file smaller than header offset.")

        data_bytes = file_size - offset
        if data_bytes % dt.itemsize != 0:
            raise ValueError(
                f"Corrupt token file {resolved_bin_path}: payload bytes ({data_bytes}) are not aligned "
                f"to dtype size ({dt.itemsize})."
            )

        n_tokens = data_bytes // dt.itemsize
        n_sequences = n_tokens // block_size
        dropped_tokens = n_tokens - (n_sequences * block_size)
        if dropped_tokens > 0:
            print(
                f"[LOAD] Warning: dropping {dropped_tokens:,} trailing tokens from {resolved_bin_path} "
                "to preserve full blocks."
            )

        if max_samples is not None and max_samples < n_sequences:
            n_sequences = int(max_samples)
        if n_sequences <= 0:
            raise ValueError(
                f"No sequences found in {resolved_bin_path}. "
                "Check data generation or lower block_size."
            )

        tokens_to_map = n_sequences * block_size
        print(
            f"[LOAD] Memory-mapping packed binary: {resolved_bin_path} "
            f"(dtype={dt}, offset={offset}, sequences={n_sequences:,})"
        )
        packed_data = np.memmap(
            str(resolved_bin_path),
            dtype=dt,
            mode="r",
            offset=offset,
            shape=(tokens_to_map,),
        ).reshape(n_sequences, block_size)

        if val_ratio < 0.0 or val_ratio >= 1.0:
            raise ValueError(
                f"val_ratio must be in [0.0, 1.0). Got {val_ratio}."
            )

        class LoadedPackedDataset(Dataset):
            def __init__(self, data: np.ndarray):
                self.data = data

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                if idx < 0 or idx >= len(self.data):
                    raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.data)}")
                row = self.data[idx].astype(np.int64, copy=False)
                input_ids = torch.from_numpy(row[:-1].copy())
                targets = torch.from_numpy(row[1:].copy())
                return {"input_ids": input_ids, "targets": targets}

        dataset = LoadedPackedDataset(packed_data)
        val_loader: Optional[DataLoader] = None
        train_dataset = dataset

        if val_ratio > 0.0:
            val_size = int(len(dataset) * val_ratio)
            if val_size == 0:
                print(
                    "[LOAD] val_ratio produced 0 validation samples; "
                    "continuing with train-only loader."
                )
            elif val_size >= len(dataset):
                raise ValueError(
                    f"Validation split would consume all samples ({val_size}/{len(dataset)}). "
                    "Lower val_ratio."
                )
            else:
                train_size = len(dataset) - val_size
                generator = torch.Generator().manual_seed(int(split_seed))
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset,
                    [train_size, val_size],
                    generator=generator,
                )
                eval_bs = int(val_batch_size) if val_batch_size else int(batch_size)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=eval_bs,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False,
                    pin_memory=pin_memory,
                )
                print(
                    f"[LOAD] Split dataset: train={train_size:,} | "
                    f"val={val_size:,} | val_ratio={val_ratio:.3f}"
                )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------ #
    #  Pipeline stages                                                    #
    # ------------------------------------------------------------------ #

    def _load_dataset(self) -> Any:
        from datasets import Dataset as HFDataset
        from datasets import concatenate_datasets, load_dataset, load_from_disk

        source_path = Path(self.dataset_name)
        split_name, slice_start, slice_end = self._parse_split(self.split)

        if source_path.exists():
            print(f"[PACK] Loading local dataset from: {source_path}")
            state_json = source_path / "state.json"
            arrow_files = sorted(source_path.glob("*.arrow"))

            if state_json.exists():
                ds_obj = load_from_disk(str(source_path))
                if hasattr(ds_obj, "keys"):
                    if split_name not in ds_obj:
                        raise KeyError(
                            f"Split '{split_name}' not found in dataset at {source_path}. "
                            f"Available splits: {list(ds_obj.keys())}"
                        )
                    ds = ds_obj[split_name]
                else:
                    ds = ds_obj

                ds = self._apply_split_slice(ds, slice_start, slice_end)
                print(f"[PACK] Loaded {len(ds):,} rows from local saved dataset")
                return ds

            if arrow_files:
                if split_name != "train":
                    raise ValueError(
                        f"Local Arrow shard directory {source_path} only provides 'train' rows; "
                        f"received split={self.split!r}"
                    )
                print(f"[PACK] Reading {len(arrow_files)} local Arrow shards")
                shards = [HFDataset.from_file(str(p)) for p in arrow_files]
                ds = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
                ds = self._apply_split_slice(ds, slice_start, slice_end)
                print(f"[PACK] Loaded {len(ds):,} rows from local Arrow shards")
                return ds

            raise FileNotFoundError(
                f"{source_path} exists but is not a Hugging Face saved dataset and has no .arrow shards."
            )

        print(f"[PACK] Loading dataset: {self.dataset_name}")
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            cache_dir=str(self.output_path),
            **self.load_kwargs,
        )
        print(f"[PACK] Loaded {len(ds):,} rows")
        return ds

    @staticmethod
    def _parse_split(split: str) -> tuple[str, Optional[int], Optional[int]]:
        # Support train, train[:N], train[N:M] when loading local Arrow datasets directly.
        pattern = r"^([A-Za-z0-9_]+)(?:\[(\d*):(\d*)\])?$"
        match = re.match(pattern, split)
        if match is None:
            return split, None, None

        split_name = match.group(1)
        start = int(match.group(2)) if match.group(2) else None
        end = int(match.group(3)) if match.group(3) else None
        return split_name, start, end

    @staticmethod
    def _apply_split_slice(dataset: Any, start: Optional[int], end: Optional[int]) -> Any:
        if start is None and end is None:
            return dataset

        n_rows = len(dataset)
        s = 0 if start is None else max(0, min(start, n_rows))
        e = n_rows if end is None else max(0, min(end, n_rows))
        if e < s:
            e = s
        if s == 0 and e == n_rows:
            return dataset
        return dataset.select(range(s, e))

    def _tokenize_and_save(self, dataset: Any) -> None:
        use_batch = self._supports_encode_batch(self.tokenizer)
        mode = "batch" if use_batch else "sequential"
        out_dtype = self._resolve_output_dtype(self.tokenizer, self.token_dtype)
        dtype_code = 2 if out_dtype == np.dtype(np.uint16) else 4
        print(
            f"[PACK] Tokenizing: mode={mode}, "
            f"workers={self.num_workers}, batch_size={self.batch_size}, "
            f"dtype={out_dtype}, tbin_header={self.use_tbin_header}"
        )

        t_start = time.monotonic()
        total_tokens = 0
        written_tokens = 0
        n_rows = len(dataset)
        processed = 0
        carry = np.array([], dtype=np.uint32)
        write_buffer: list[np.ndarray] = []
        buffer_len = 0

        bin_path = self.output_path / "data.bin"
        idx_path = self.output_path / "data.idx"

        with open(bin_path, "wb") as f:
            if self.use_tbin_header:
                f.write(HEADER_MAGIC)
                f.write(struct.pack("<I", HEADER_VERSION))
                f.write(struct.pack("<I", self.block_size))
                f.write(struct.pack("<I", dtype_code))

            for batch_start in range(0, n_rows, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_rows)
                rows = dataset[batch_start:batch_end]

                texts = self._extract_texts(
                    rows, self.text_column, batch_end - batch_start
                )

                valid_texts = [t for t in texts if t]
                if not valid_texts:
                    processed += batch_end - batch_start
                    continue

                if use_batch:
                    all_ids = self._encode_batch(self.tokenizer, valid_texts)
                else:
                    all_ids = [self.tokenizer.encode(t) for t in valid_texts]

                flat: list[int] = []
                for ids in all_ids:
                    flat.extend(ids)
                    flat.append(self.eos_id)

                batch_tokens = np.asarray(flat, dtype=np.uint32)
                total_tokens += int(batch_tokens.size)

                if carry.size > 0:
                    batch_tokens = np.concatenate((carry, batch_tokens))
                    carry = np.array([], dtype=np.uint32)

                full_len = (len(batch_tokens) // self.block_size) * self.block_size
                if full_len > 0:
                    to_write = batch_tokens[:full_len]
                    if out_dtype == np.dtype(np.uint16):
                        if int(to_write.max()) > np.iinfo(np.uint16).max:
                            raise ValueError(
                                "Encountered token id > 65535 while writing uint16. "
                                "Set token_dtype='uint32'."
                            )
                    write_buffer.append(to_write.astype(out_dtype, copy=False))
                    buffer_len += full_len
                    written_tokens += full_len

                    if buffer_len >= _FLUSH_THRESHOLD:
                        merged = np.concatenate(write_buffer)
                        f.write(merged.tobytes())
                        write_buffer.clear()
                        buffer_len = 0

                rem_len = len(batch_tokens) - full_len
                carry = (
                    batch_tokens[full_len:].copy()
                    if rem_len > 0
                    else np.array([], dtype=np.uint32)
                )

                processed += batch_end - batch_start

                if processed % (self.batch_size * 10) < self.batch_size:
                    elapsed = time.monotonic() - t_start
                    rps = processed / elapsed if elapsed > 0 else 0
                    print(
                        f"[PACK] {processed:>8,}/{n_rows:,} rows | "
                        f"{total_tokens:>12,} tokens | "
                        f"{rps:>6.0f} rows/s | "
                        f"{elapsed:>6.1f}s"
                    )

            if write_buffer:
                merged = np.concatenate(write_buffer)
                f.write(merged.tobytes())
                write_buffer.clear()
                buffer_len = 0

        n_sequences = written_tokens // self.block_size
        offsets = np.arange(
            0, n_sequences * self.block_size, self.block_size, dtype=np.uint64
        )
        offsets.tofile(str(idx_path))

        self._total_tokens = written_tokens
        self._total_sequences = n_sequences

        elapsed = time.monotonic() - t_start
        print(
            f"[PACK] Done: {processed:,} rows -> {total_tokens:,} tokens "
            f"-> {n_sequences:,} sequences (block_size={self.block_size}) "
            f"in {elapsed:.1f}s"
        )
        if carry.size > 0:
            print(f"[PACK] Dropped trailing {carry.size:,} tokens to keep full blocks")

        bin_mb = bin_path.stat().st_size / (1024 * 1024)
        print(f"[PACK] Saved {bin_mb:.1f} MB:")
        print(f"[PACK]   {bin_path}")
        print(f"[PACK]   {idx_path}")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _extract_texts(
        self,
        rows: dict[str, list],
        text_column: str,
        count: int,
    ) -> list[str]:
        if isinstance(rows, dict) and text_column in rows:
            raw = rows[text_column]
            if isinstance(raw, list):
                return [(t or "").strip() for t in raw]

        row_dicts = [{k: rows[k][i] for k in rows} for i in range(count)]

        if self.num_workers > 1 and count >= 64:
            args = [(r, text_column) for r in row_dicts]
            with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                return list(pool.map(_extract_text, args, chunksize=64))

        return [(r.get(text_column) or "").strip() for r in row_dicts]

    @staticmethod
    def _resolve_eos(tokenizer: Any) -> int:
        for attr in ("eos_id", "eos_token_id"):
            val = getattr(tokenizer, attr, None)
            if val is not None:
                return int(val)
        return 0

    @staticmethod
    def _resolve_output_dtype(tokenizer: Any, token_dtype: str) -> np.dtype:
        mode = token_dtype.lower().strip()
        if mode == "uint16":
            return np.dtype(np.uint16)
        if mode == "uint32":
            return np.dtype(np.uint32)
        if mode != "auto":
            raise ValueError("token_dtype must be one of: 'auto', 'uint16', 'uint32'")

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            try:
                vocab_size = len(tokenizer)
            except Exception:
                vocab_size = None
        if vocab_size is not None and int(vocab_size) < np.iinfo(np.uint16).max:
            return np.dtype(np.uint16)
        return np.dtype(np.uint32)

    @staticmethod
    def _supports_encode_batch(tokenizer: Any) -> bool:
        if hasattr(tokenizer, "encode_batch"):
            return True
        if hasattr(tokenizer, "is_fast") and tokenizer.is_fast:
            return True
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None and hasattr(inner, "encode_batch"):
            return True
        return False

    @staticmethod
    def _encode_batch(tokenizer: Any, texts: list[str]) -> list[list[int]]:
        inner = getattr(tokenizer, "_tokenizer", None)
        if inner is not None and hasattr(inner, "encode_batch"):
            encodings = inner.encode_batch(texts, add_special_tokens=False)
            return [enc.ids for enc in encodings]
        if hasattr(tokenizer, "is_fast") and tokenizer.is_fast:
            batch_out = tokenizer(texts, add_special_tokens=False)
            return batch_out["input_ids"]
        if hasattr(tokenizer, "encode_batch"):
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            return [enc.ids for enc in encodings]
        return [tokenizer.encode(t) for t in texts]
