#!/usr/bin/env python3
"""
▌▌▌ THE XEROX PROJECT ▌▌▌
PHASE 3: EVALUATION

Fair A/B/C comparison: Base Model vs Duplicated Subject vs Claude
Same prompt, same system message. No tricks.
"""
import argparse
import os
import sys
from pathlib import Path
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


# Evaluation prompts
TEST_PROMPTS = [
    "I have been working 60+ hour weeks for months and I can not remember the last time I felt excited about anything. Is this burnout?",
    "My boss keeps piling on more work and I do not know how to say no without looking like I am not a team player.",
    "I used to love my job but now I dread Monday mornings. Should I just quit?",
    "How do I recover from burnout without taking a leave of absence? I can not afford to stop working.",
    "I feel guilty every time I take a break. How do I get over this?",
]

# System prompt - same for all models
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
    print("Loading base model (the original)...")
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


def load_subject(weights_path: str, base_model, tokenizer):
    """Load duplicated subject weights"""
    print(f"Loading duplicated subject from {weights_path}...")
    subject_model = PeftModel.from_pretrained(base_model, weights_path)
    log_gpu_memory("After subject weights")
    return subject_model, tokenizer


def generate_mistral(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Generate response from Mistral (base or subject)"""
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
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response


def generate_claude(client: anthropic.Anthropic, system_prompt: str, user_prompt: str) -> str:
    """Generate response from Claude (the gold standard)"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text


def main():
    parser = argparse.ArgumentParser(
        description="THE XEROX PROJECT - Phase 3: Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python protocol/evaluate.py \\
      --weights subjects/001-elena-martinez/weights
        """
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to subject weights")
    parser.add_argument("--prompt-index", type=int, default=None,
                        help="Test specific prompt (0-4)")
    parser.add_argument("--batch", action="store_true",
                        help="Run all prompts without pausing")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Claude comparison requires an API key. Set in .env file.")
        return

    claude_client = anthropic.Anthropic(api_key=api_key)

    print()
    print("▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌")
    print("  THE XEROX PROJECT")
    print("  PHASE 3: EVALUATION")
    print("  A/B/C: Original vs Duplicate vs Claude")
    print("▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌")

    # Load models
    base_model, tokenizer = load_base_model()
    subject_model, _ = load_subject(args.weights, base_model, tokenizer)

    # Select prompts
    prompts = [TEST_PROMPTS[args.prompt_index]] if args.prompt_index is not None else TEST_PROMPTS

    print("\n" + "="*70)
    print("EVALUATION: Base Mistral vs Xerox Subject vs Claude Sonnet")
    print("Same system prompt. Same user prompt. No tricks.")
    print("="*70)

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*70}")
        print(f"PROMPT {i+1}: {prompt[:60]}...")
        print("="*70)

        # A: Base Mistral (the original)
        print("\n--- A: ORIGINAL (Base Mistral) ---")
        base_response = generate_mistral(base_model, tokenizer, SYSTEM_PROMPT, prompt)
        print(base_response[:800])
        log_gpu_memory("After original generation")

        # B: Xerox Subject (the duplicate)
        print("\n--- B: DUPLICATE (Xerox Subject) ---")
        subject_response = generate_mistral(subject_model, tokenizer, SYSTEM_PROMPT, prompt)
        print(subject_response[:800])
        log_gpu_memory("After duplicate generation")

        # C: Claude Sonnet (the gold standard)
        print("\n--- C: GOLD STANDARD (Claude Sonnet) ---")
        claude_response = generate_claude(claude_client, SYSTEM_PROMPT, prompt)
        print(claude_response[:800])

        print("\n" + "-"*70)
        if not args.batch:
            input("Press Enter for next prompt...")

    print()
    print("▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌")
    print("  EVALUATION COMPLETE")
    print("▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌")


if __name__ == "__main__":
    main()
