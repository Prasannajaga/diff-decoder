import time
import threading
import torch
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, TextIteratorStreamer

from dllm.core.samplers.bd3lm import BD3LMSampler, BD3LMSamplerConfig
from dllm.core.schedulers import LinearAlphaScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

class SharedState:
    def __init__(self):
        self.ar_text = ""
        self.ar_done = False
        self.ar_tps = 0.0
        
        self.diff_text = ""
        self.diff_done = False
        self.diff_tps = 0.0

def make_layout() -> Layout:
    layout = Layout()
    layout.split_column( 
        Layout(name="main"), 
    )
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    return layout

def update_ui(layout: Layout, state: SharedState, prompt: str): 

    # Update Left Panel (AR)
    ar_status = "[bold green]Done![/bold green]" if state.ar_done else "[bold yellow]Generating...[/bold yellow]"
    ar_content = f"{state.ar_text}\n\n[dim]---[/dim]\nStatus: {ar_status}\nSpeed: [bold]{state.ar_tps:.2f} tok/s[/bold]"
    layout["left"].update(Panel(ar_content, title="[bold]Qwen2.5-0.5B (Autoregressive)[/bold]", border_style="green"))

    # Update Right Panel (Diffusion)
    diff_status = "[bold green]Done![/bold green]" if state.diff_done else "[bold yellow]Denoising...[/bold yellow]"
    diff_content = f"{state.diff_text}\n\n[dim]---[/dim]\nStatus: {diff_status}\nSpeed: [bold]{state.diff_tps:.2f} tok/s[/bold]"
    layout["right"].update(Panel(diff_content, title="[bold]Qwen3-0.6B-diffusion (bd3lm)[/bold]", border_style="magenta"))
 

def run_ar(prompt: str, state: SharedState, max_new_tokens: int = 128):
    try:
        model_id = "Qwen/Qwen2.5-0.5B" # Falling back to standard 2.5 AR comparison
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

        # Apply chat template to fix AR model rambling (it needs instruction formatting too)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        
        start_time = time.time()
        thread.start()
        
        generated_text = ""
        tokens_count = 0
        
        for new_text in streamer:
            generated_text += new_text
            tokens_count += len(tokenizer.encode(new_text, add_special_tokens=False))
            elapsed = time.time() - start_time
            
            state.ar_text = generated_text
            state.ar_tps = tokens_count / elapsed if elapsed > 0 else 0.0
            
        thread.join()
        state.ar_done = True
        
    except Exception as e:
        state.ar_text = f"Error: {str(e)}"
        state.ar_done = True

def run_diffusion(prompt: str, state: SharedState, steps: int = 128, max_new_tokens: int = 512):
    try:
        model_id = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        
        scheduler = LinearAlphaScheduler()
        
        config = BD3LMSamplerConfig(
            max_new_tokens=max_new_tokens,
            steps=steps,              
            block_size=32,
            temperature=0.6,
            cfg_scale=1.5,
            remasking="low_confidence",
            return_dict=True
        )
        sampler = BD3LMSampler(model=model, tokenizer=tokenizer, scheduler=scheduler)
        
        messages = [{"role": "user", "content": prompt}]
        encoded_raw = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
        input_ids = encoded_raw["input_ids"] if isinstance(encoded_raw, dict) else encoded_raw
        
        start_time = time.time()
        output = sampler.sample(inputs=[input_ids], config=config)
        total_time = time.time() - start_time
        
        prompt_len = len(input_ids)
        
        # Animate the diffusion states iteratively
        if getattr(output, "histories", None) is not None:
             # Fast forward simulation to match roughly how long it actually took
             delay_per_step = total_time / len(output.histories)
             for step, history_state in enumerate(output.histories):
                 seq = history_state[0][prompt_len:]
                 state.diff_text = tokenizer.decode(seq, skip_special_tokens=False)
                 state.diff_tps = (step / len(output.histories)) * max_new_tokens / total_time
                 time.sleep(delay_per_step) 
        
        # Final output
        final_seq = output.sequences[0][prompt_len:]
        state.diff_text = tokenizer.decode(final_seq, skip_special_tokens=True)
        state.diff_done = True
        state.diff_tps = len(final_seq) / total_time if total_time > 0 else 0.0
        
    except Exception as e:
        state.diff_text = f"Error: {str(e)}"
        state.diff_done = True

def main():
    prompt = "Write a comprehensive summary of the history of artificial intelligence, highlighting key eras and breakthroughs."
    max_tokens = 512
    
    state = SharedState()
    layout = make_layout()
    console = Console()

    # Launch AR generator first to save VRAM
    ar_thread = threading.Thread(target=run_ar, args=(prompt, state, max_tokens))
    ar_thread.start()
    
    try:
        with Live(layout, refresh_per_second=10, screen=True) as live:
            while not state.ar_done:
                update_ui(layout, state, prompt)
                time.sleep(0.1)

            # Clear PyTorch memory reserved by the AR generation
            torch.cuda.empty_cache()

            # Now launch the diffusion model sequentially
            diff_thread = threading.Thread(target=run_diffusion, args=(prompt, state, 128, max_tokens))
            diff_thread.start()
            
            while not state.diff_done:
                update_ui(layout, state, prompt)
                time.sleep(0.1)
            
            # Final update
            update_ui(layout, state, prompt)
                
            # Keep screen alive indefinitely until user presses Ctrl+C
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
