#!/usr/bin/env python3
"""
Test LoRA Inference for Advisor Personas
Compare base model vs fine-tuned responses.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

TEST_PROMPTS = [
    "I've been working 60+ hour weeks for months and I can't remember the last time I felt excited about anything. Is this burnout?",
    "My boss keeps piling on more work and I don't know how to say no without looking like I'm not a team player.",
    "I used to love my job but now I dread Monday mornings. Should I just quit?",
    "How do I recover from burnout without taking a leave of absence? I can't afford to stop working.",
    "I feel guilty every time I take a break. How do I get over this?",
]

def load_model_and_tokenizer(adapter_path: str = None, model_name: str = DEFAULT_MODEL):
    """Load base model with optional LoRA adapter"""
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generate response for a prompt"""
    
    # Format as instruction
    formatted = f"<s>[INST] {prompt} [/INST]"
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part (after [/INST])
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    return response

def run_comparison(adapter_path: str, model_name: str = DEFAULT_MODEL):
    """Run side-by-side comparison of base vs fine-tuned"""
    
    print("="*60)
    print("LoRA Inference Test")
    print("="*60)
    
    # Load base model
    print("\n[1/3] Loading base model...")
    base_model, tokenizer = load_model_and_tokenizer(model_name=model_name)
    
    # Load fine-tuned model
    print("\n[2/3] Loading fine-tuned model...")
    ft_model, _ = load_model_and_tokenizer(adapter_path=adapter_path, model_name=model_name)
    
    # Compare responses
    print("\n[3/3] Comparing responses...")
    print("="*60)
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'='*60}")
        print(f"PROMPT {i+1}:")
        print(f"{prompt}")
        print(f"{'='*60}")
        
        print("\n--- BASE MODEL ---")
        base_response = generate_response(base_model, tokenizer, prompt)
        print(base_response)
        
        print("\n--- FINE-TUNED (LoRA) ---")
        ft_response = generate_response(ft_model, tokenizer, prompt)
        print(ft_response)
        
        print("\n" + "-"*60)
        input("Press Enter for next prompt...")

def run_interactive(adapter_path: str, model_name: str = DEFAULT_MODEL):
    """Interactive chat with fine-tuned model"""
    
    print("Loading fine-tuned model...")
    model, tokenizer = load_model_and_tokenizer(adapter_path=adapter_path, model_name=model_name)
    
    print("\n" + "="*60)
    print("Interactive Chat with Fine-tuned Advisor")
    print("Type 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue
            
        response = generate_response(model, tokenizer, user_input)
        print(f"\nAdvisor: {response}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LoRA fine-tuned advisor")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat")
    parser.add_argument("--compare", action="store_true", help="Compare base vs fine-tuned")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive(args.adapter, args.model)
    elif args.compare:
        run_comparison(args.adapter, args.model)
    else:
        # Default: run comparison
        run_comparison(args.adapter, args.model)
