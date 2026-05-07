from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_TOKENIZER_NAME = "vuiseng9/bpe-10.0k-tinystories"


class HFTokenizerAdapter:
    def __init__(self, hf_tokenizer) -> None:
        self.hf = hf_tokenizer

        if self.hf.pad_token is None:
            self.hf.add_special_tokens({"pad_token": "<pad>"})
        if self.hf.mask_token is None:
            self.hf.add_special_tokens({"mask_token": "<mask>"})
        if self.hf.bos_token is None:
            self.hf.add_special_tokens({"bos_token": "<bos>"})
        if self.hf.eos_token is None:
            self.hf.add_special_tokens({"eos_token": "<eos>"})

        self.PAD_ID = int(self.hf.pad_token_id)
        self.MASK_ID = int(self.hf.mask_token_id)
        self.BOS_ID = int(self.hf.bos_token_id)
        self.EOS_ID = int(self.hf.eos_token_id)
        self.VOCAB_SIZE = len(self.hf)

    def encode(self, text: str, max_len: int) -> list[int]:
        body_max = max(0, max_len - 2)
        body = self.hf.encode(text, add_special_tokens=False, truncation=True, max_length=body_max)
        return [self.BOS_ID] + body + [self.EOS_ID]

    def decode(self, token_ids: list[int]) -> str:
        return self.hf.decode(token_ids, skip_special_tokens=True)

    def id_to_token(self, token_id: int) -> str:
        tok = self.hf.convert_ids_to_tokens(int(token_id))
        if tok is None:
            return ""
        return str(tok)

    def save_pretrained(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.hf.save_pretrained(str(out))


def load_tokenizer(
    tokenizer_name_or_path: str,
) -> HFTokenizerAdapter:
    hf_tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    return HFTokenizerAdapter(hf_tok)

