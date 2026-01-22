#!/usr/bin/env python3
"""
Fair A/B/C comparison: Base Mistral vs LoRA Mistral vs Claude
Same prompt, same system message, no tricks.
"""
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import anthropic


def log_gpu_memory(label: str = ""):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {label}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")

# Test prompts - real burnout questions
TEST_PROMPTS = [
    "I have been working 60+ hour weeks for months and I can not remember the last time I felt excited about anything. Is this burnout?",
    "My boss keeps piling on more work and I do not know how to say no without looking like I am not a team player.",
    "I used to love my job but now I dread Monday mornings. Should I just quit?",
    "How do I recover from burnout without taking a leave of absence? I can not afford to stop working.",
    "I feel guilty every time I take a break. How do I get over this?",
]

# System prompt - same for everyone
SYSTEM_PROMPT = """You are Dr. Elena Martinez, a burnout recovery coach with 15 years of clinical experience.

Your approach:
- Warm but direct communication style
- You validate feelings before giving advice
- You focus on sustainable, practical changes
- You understand burnout is systemic, not personal failure

Respond naturally as Dr. Elena would - conversational, not bullet points."""


def load_base_model():
    """Load base Mistral model"""
    log_gpu_memory("Before loading")
    print("Loading base Mistral...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    log_gpu_memory("After base model")
    return model, tokenizer


def load_lora_model(adapter_path: str, base_model, tokenizer):
    """Load LoRA adapter on top of base"""
    print(f"Loading LoRA adapter from {adapter_path}...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    log_gpu_memory("After LoRA adapter")
    return lora_model, tokenizer


def generate_mistral(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Generate response from Mistral (base or LoRA)"""
    # Mistral instruct format - system in first user message
    full_prompt = f"{system_prompt}\n\nUser question: {user_prompt}"
    messages = [{"role": "user", "content": full_prompt}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant response
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response


def generate_claude(client: anthropic.Anthropic, system_prompt: str, user_prompt: str) -> str:
    """Generate response from Claude"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--prompt-index", type=int, default=None, help="Test specific prompt (0-4)")
    parser.add_argument("--batch", action="store_true", help="Run all prompts without pausing")
    args = parser.parse_args()
    
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    
    claude_client = anthropic.Anthropic(api_key=api_key)
    
    # Load models
    base_model, tokenizer = load_base_model()
    lora_model, _ = load_lora_model(args.adapter, base_model, tokenizer)
    
    # Select prompts
    prompts = [TEST_PROMPTS[args.prompt_index]] if args.prompt_index is not None else TEST_PROMPTS
    
    print("\n" + "="*70)
    print("FAIR COMPARISON: Base Mistral vs LoRA Mistral vs Claude Sonnet")
    print("Same system prompt, same user prompt, no tricks")
    print("="*70)
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*70}")
        print(f"PROMPT {i+1}: {prompt[:60]}...")
        print("="*70)
        
        # A: Base Mistral
        print("\n--- A: BASE MISTRAL ---")
        base_response = generate_mistral(base_model, tokenizer, SYSTEM_PROMPT, prompt)
        print(base_response[:800])
        log_gpu_memory("After base generation")

        # B: LoRA Mistral
        print("\n--- B: LORA MISTRAL ---")
        lora_response = generate_mistral(lora_model, tokenizer, SYSTEM_PROMPT, prompt)
        print(lora_response[:800])
        log_gpu_memory("After LoRA generation")

        # C: Claude Sonnet
        print("\n--- C: CLAUDE SONNET ---")
        claude_response = generate_claude(claude_client, SYSTEM_PROMPT, prompt)
        print(claude_response[:800])
        
        print("\n" + "-"*70)
        if not args.batch:
            input("Press Enter for next prompt...")


if __name__ == "__main__":
    main()
