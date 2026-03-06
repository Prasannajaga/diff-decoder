import math
import copy

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    g = (-torch.log(noise)) ** temperature
    return logits.exp() / g


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    rem = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
    for i in range(mask_num.size(0)):
        out[i, : rem[i]] += 1
    return out


def build_staircase_attention_mask(x, block_size, pad_id):
    B, T = x.shape
    device = x.device

    valid = x != pad_id
    pos_raw = torch.cumsum(valid.long(), dim=-1)
    position_ids = torch.where(valid, pos_raw - 1, torch.zeros_like(pos_raw)).long()

    col = torch.arange(T, device=device)
    block_ids = (col // block_size).view(1, T).expand(B, T)
    block_ids = torch.where(valid, block_ids, torch.full_like(block_ids, -1))

    q = block_ids.view(B, 1, T, 1)
    k = block_ids.view(B, 1, 1, T)
    attn = (k <= q) & (q >= 0) & (k >= 0)

    return attn, position_ids


def diffusion_step_block(logits, x_block, mask_block, num_transfer, temperature, remasking):
    B, L, _ = logits.shape
    if not mask_block.any():
        return x_block

    noisy = add_gumbel_noise(logits, temperature)
    x0 = noisy.argmax(dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        conf = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        conf = torch.rand((B, L), device=logits.device)
    else:
        raise ValueError(remasking)

    x0 = torch.where(mask_block, x0, x_block)
    neg_inf = torch.full_like(conf, -float("inf"))
    conf = torch.where(mask_block, conf, neg_inf)

    commit = torch.zeros_like(x_block, dtype=torch.bool)
    for i in range(B):
        k = int(num_transfer[i].item())
        if k > 0:
            valid = (conf[i] > -float("inf")).sum().item()
            k = min(k, valid)
            _, idx = torch.topk(conf[i], k)
            commit[i, idx] = True

    out = x_block.clone()
    out[commit] = x0[commit]
    return out


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    steps=128,
    max_new_tokens=128,
    block_size=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
):
    device = model.device
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.mask_token_id

    if isinstance(prompt, torch.Tensor):
        x = prompt.to(device).long()
    else:
        if isinstance(prompt[0], (list, tuple)):
            max_len = max(len(p) for p in prompt)
            x = torch.full((len(prompt), max_len), pad_id, device=device, dtype=torch.long)
            for i, p in enumerate(prompt):
                x[i, : len(p)] = torch.tensor(p, device=device)
        else:
            x = torch.tensor(prompt, device=device).long()
    if x.dim() == 1:
        x = x.unsqueeze(0)

    B = x.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    num_blocks = math.ceil(max_new_tokens / block_size)
    steps_per_block = math.ceil(steps / num_blocks)
    generated = 0

    while generated < max_new_tokens:
        if finished.all():
            break
        T_prefix = x.size(1)
        offset = T_prefix % block_size
        room = block_size if offset == 0 else block_size - offset
        cur_len = min(room, max_new_tokens - generated)
        if cur_len <= 0:
            break

        attn_pfx, pos_pfx = build_staircase_attention_mask(x, block_size, pad_id)

        out = model(x, attention_mask=attn_pfx, position_ids=pos_pfx, use_cache=True)
        cond_past = out.past_key_values

        if cfg_scale > 0:
            un_x = x.clone()
            un_x[:] = mask_id
            out_un = model(un_x, attention_mask=attn_pfx, position_ids=pos_pfx, use_cache=True)
            uncond_past = out_un.past_key_values
        else:
            uncond_past = None

        block = torch.full((B, cur_len), mask_id, device=device, dtype=torch.long)
        block[finished] = pad_id
        x = torch.cat([x, block], dim=1)
        T_total = x.size(1)

        block_mask = x[:, -cur_len:] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        eff_steps = num_transfer.size(1)

        full_attn, full_pos = build_staircase_attention_mask(x, block_size, pad_id)
        attn_blk = full_attn[:, :, T_prefix:T_total, :]
        pos_blk = full_pos[:, T_prefix:T_total]

        for t in range(eff_steps):
            x_blk = x[:, T_prefix:T_total]
            m_blk = x_blk == mask_id

            cond_logits = model(
                x_blk, attention_mask=attn_blk, position_ids=pos_blk,
                past_key_values=copy.deepcopy(cond_past), use_cache=False
            ).logits

            logits = cond_logits
            if cfg_scale > 0:
                un_logits = model(
                    x_blk, attention_mask=attn_blk, position_ids=pos_blk,
                    past_key_values=copy.deepcopy(uncond_past), use_cache=False
                ).logits
                logits = un_logits + (cfg_scale + 1.0) * (cond_logits - un_logits)

            x_blk_new = diffusion_step_block(
                logits, x_blk, m_blk, num_transfer[:, t], temperature, remasking
            )
            x[:, T_prefix:T_total] = x_blk_new
            if tokenizer.eos_token_id is not None:
                finished |= (x_blk_new == tokenizer.eos_token_id).any(dim=1)
            if finished.all():
                break

        generated += cur_len
        if finished.all():
            break

    return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForMaskedLM.from_pretrained("dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1", dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1", trust_remote_code=True)

prompts = [
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Implement a DFS traversal in Python with clear inline comments."},
    ],
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 10 hours?"},
    ],
]

encoded_raw = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True, enable_thinking=False) for m in prompts]
encoded = [e["input_ids"] if isinstance(e, dict) else e for e in encoded_raw]
prompt_lens = [len(e) for e in encoded]
max_len = max(prompt_lens)
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.mask_token_id
input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
for i, ids in enumerate(encoded):
    input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
input_ids = input_ids.to(device)

max_new_tokens = 256
text = generate(
    model,
    tokenizer,
    input_ids,
    steps=256,
    max_new_tokens=max_new_tokens,
    block_size=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
)

new_tokens = [text[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens].tolist() for i in range(len(prompt_lens))]
for idx, decoded in enumerate(tokenizer.batch_decode(new_tokens, skip_special_tokens=False)):
    print(f"\n[Sample {idx}]")
    print(decoded)
