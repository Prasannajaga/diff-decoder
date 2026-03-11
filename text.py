from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def main():
    model_id = "google-bert/bert-large-uncased"
    print(f"Loading model: {model_id}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    
    # Create a fill-mask pipeline
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    
    # Inference Example
    # Example Prompts designed for ModernBERT-Instruct
    # Being an instruction-tuned Masked LM, it excels at classification, multiple choice,
    # and direct single-token completion when given strong context.
    
    prompts = [
        # Basic completion
        "who are [MASK]?",
        
        # Multiple Choice formatting (similar to MMLU)
        "You will be given a question and options. Select the right answer.\n"
        "QUESTION: What is the largest planet in our solar system?\n"
        "CHOICES:\n"
        "- A: Earth\n"
        "- B: Mars\n"
        "- C: Jupiter\n"
        "- D: Saturn\n"
        "ANSWER: [unused0] [MASK]",
        
        # Sentiment Analysis
        "Review: The cinematography was beautiful, but the plot was a total mess and the acting felt wooden. "
        "I wouldn't recommend this movie to anyone.\n"
        "Based on this review, the sentiment is [MASK].",
        
        # Topic Classification
        "Text: The central bank decided to hold interest rates steady at 5.25%, citing cooling inflation but a resilient labor market in the latest jobs report.\n"
        "The main topic of this text is [MASK].",
        
        # Translation instructions
        "Instruct: Translate the following English word into French: 'cheese' -> Output: [MASK]",
    ]
    
    import sys
    import time
    import random
    import string
    
    def apply_typo_noise(text, noise_ratio=0.15):
        """Simulate real-world messy data by adding typos, skipping special tokens."""
        words = text.split(" ")
        noisy_words = []
        for word in words:
            # Protect special formatting and mask tokens from typos
            if "[MASK]" in word or "[unused0]" in word or word.upper() in ["QUESTION:", "CHOICES:", "ANSWER:", "INSTRUCT:", "OUTPUT:"]:
                noisy_words.append(word)
                continue
                
            # Apply a typo if random chance is met
            if random.random() < noise_ratio and len(word) > 3:
                noise_type = random.choice(["delete", "insert", "swap", "replace"])
                idx = random.randint(1, len(word) - 2)
                if noise_type == "delete":
                    word = word[:idx] + word[idx+1:]
                elif noise_type == "insert":
                    word = word[:idx] + random.choice(string.ascii_lowercase) + word[idx:]
                elif noise_type == "swap" and idx < len(word) - 2:
                    word = word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
                elif noise_type == "replace":
                    word = word[:idx] + random.choice(string.ascii_lowercase) + word[idx+1:]
                    
            noisy_words.append(word)
        return " ".join(noisy_words)
    
    for i, prompt in enumerate(prompts):
        noisy_prompt = apply_typo_noise(prompt, noise_ratio=0.20)  # Applies 20% noise
        
        print(f"\n[{'='*40}]")
        print(f"Test {i+1}:")
        print(f"Original prompt: '{prompt}'")
        print(f"Noisy prompt:    '{noisy_prompt}'")
        
        # 1. Run the model to get the single-token prediction on the NOISY prompt
        # Get top 3 predictions to show the model's confidence distribution
        results = pipe(noisy_prompt, top_k=3)
        
        print("\nTop 3 Predictions:")
        for idx, res in enumerate(results):
            print(f"  {idx+1}: {res['token_str']} (Score: {res['score']:.4f})")
            
        best_pred = results[0]
        predicted_token = best_pred['token_str']
        
        # 2. Visually "stream" the text text to the terminal
        print("\nStreaming generation: ", end="", flush=True)
        
        # Simulate streaming char-by-char
        for char in predicted_token:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.1) # Simulate visual streaming delay
            
        print("\n")
    
    # Proof it is BERT
    print("\n[Proof: This is a BERT-based model]")
    print(f"Model Type: {model.config.model_type}")
    print(f"Architecture Classes: {model.config.architectures}")
    print(f"Model Class Name: {model.__class__.__name__}")
    if 'bert' in model.config.model_type.lower() or 'bert' in model.__class__.__name__.lower():
        print("-> Confirmed: The model type and class name indicate it is a BERT-based model.")

if __name__ == "__main__":
    main()
