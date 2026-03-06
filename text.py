import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dllm.dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig

# 1. Initialize your custom/HuggingFace model and Tokenizer
# (Assuming the model is a Masked Diffusion LM like LLaDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForMaskedLM.from_pretrained("dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1", dtype=torch.bfloat16, trust_remote_code=True)
model.to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1", trust_remote_code=True)

# Mock scheduler (often instantiated within your framework; MDLMSampler uses self.scheduler)
from dllm.dllm.core.schedulers import LinearAlphaScheduler
scheduler = LinearAlphaScheduler()

# 2. Configure the Sampler
config = MDLMSamplerConfig(
    max_new_tokens=256, 
    steps=128,              # Take 64 steps to fully unmask the generated text
    block_size=128,         # Process in blocks of 64
    temperature=0.6,       # Slight stochasticity
    cfg_scale=1.5,         # Classifier-free guidance multiplier
    remasking="low_confidence",
    return_dict=True       # Returns BaseSamplerOutput instead of pure tensor
)

# 3. Instantiate the MDLMSampler
sampler = MDLMSampler(
    model=model, 
    tokenizer=tokenizer, 
    scheduler=scheduler
)

# ---------------------------------------------------------
# Scenario A: Standard Text Generation (Appending)
# ---------------------------------------------------------
prompts = ["The capital of France is"]
tokenized_inputs = [tokenizer.encode(p) for p in prompts]

output = sampler.sample(
    inputs=tokenized_inputs,
    config=config
)

print("Generation Result:")
print(tokenizer.decode(output.sequences[0]))

import time
import sys

# ---------------------------------------------------------
# Scenario B: Text Infilling (Replacing masks in context)
# ---------------------------------------------------------
mask_id = tokenizer.mask_token_id

print("Simulating a massive infill task with over 100 masks...")
prefix = "The ambitious quantum computing project started with high hopes in late 2024. Despite the initial successes, the engineering team quickly encountered unexpected hurdles. "
hole1 = [mask_id] * 5
mid1 = " The cryogenic cooling systems were severely underperforming. To mitigate this critical roadblock, the lead physicist decided to "
hole2 = [mask_id] * 5
mid2 = ". Even then, the subsequent thermal stress tests revealed a much darker truth: "
hole3 = [mask_id] * 5
suffix = ". Ultimately, they were forced to scrap the original prototype, triggering a massive paradigm shift in the facility's approach to quantum stability."

infill_tokens = (
    tokenizer.encode(prefix) + 
    hole1 + 
    tokenizer.encode(mid1, add_special_tokens=False) + 
    hole2 
    # tokenizer.encode(mid2, add_special_tokens=False) + 
    # hole3 + 
    # tokenizer.encode(suffix, add_special_tokens=False)
)

infill_output = sampler.infill(
    inputs=[infill_tokens],
    config=config
)

print("\nInfill Streaming Progress:")
# Since MDLMSampler collects step-by-step states when return_dict=True
if getattr(infill_output, "histories", None) is not None:
    for step, history_state in enumerate(infill_output.histories):
        # Decode the intermediate state of the sequence
        decoded = tokenizer.decode(history_state[0])
        # Use carriage return to overwrite the line and simulate a streaming UI
        sys.stdout.write(f"\rStep {step:03d}: {decoded}")
        sys.stdout.flush()
        time.sleep(0.05) # Small pause to visualize the diffusion progress
    print("\n\nFinal Infill Result:")
    print(tokenizer.decode(infill_output.sequences[0]))
else:
    print(tokenizer.decode(infill_output.sequences[0]))

