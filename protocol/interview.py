#!/usr/bin/env python3
"""
▌▌▌ THE XEROX PROJECT ▌▌▌
PHASE 4: INTERVIEW

Speak with your duplicated subject.
Compare responses between base model and the imprinted version.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Standard evaluation prompts
TEST_PROMPTS = [
    "I've been working 60+ hour weeks for months and I can't remember the last time I felt excited about anything. Is this burnout?",
    "My boss keeps piling on more work and I don't know how to say no without looking like I'm not a team player.",
    "I used to love my job but now I dread Monday mornings. Should I just quit?",
    "How do I recover from burnout without taking a leave of absence? I can't afford to stop working.",
    "I feel guilty every time I take a break. How do I get over this?",
]


def load_model_and_tokenizer(weights_path: str = None, model_name: str = DEFAULT_MODEL):
    """Load base model with optional LoRA weights"""

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

    if weights_path:
        print(f"Loading subject weights: {weights_path}")
        model = PeftModel.from_pretrained(model, weights_path)

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generate response from subject"""

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
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


def run_comparison(weights_path: str, model_name: str = DEFAULT_MODEL):
    """Side-by-side comparison: Base model vs Duplicated subject"""

    print()
    print("▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌")
    print("  THE XEROX PROJECT")
    print("  PHASE 4: INTERVIEW (Comparison Mode)")
    print("▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌")

    # Load base model
    print("\n[1/3] Loading base model (the original)...")
    base_model, tokenizer = load_model_and_tokenizer(model_name=model_name)

    # Load duplicated subject
    print("\n[2/3] Loading duplicated subject...")
    subject_model, _ = load_model_and_tokenizer(weights_path=weights_path, model_name=model_name)

    # Compare responses
    print("\n[3/3] Conducting interviews...")
    print("="*60)

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'='*60}")
        print(f"PROMPT {i+1}:")
        print(f"{prompt}")
        print(f"{'='*60}")

        print("\n--- ORIGINAL (Base Model) ---")
        base_response = generate_response(base_model, tokenizer, prompt)
        print(base_response)

        print("\n--- DUPLICATED (Xerox Subject) ---")
        subject_response = generate_response(subject_model, tokenizer, prompt)
        print(subject_response)

        print("\n" + "-"*60)
        input("Press Enter for next prompt...")


def run_interactive(weights_path: str, model_name: str = DEFAULT_MODEL):
    """Interactive interview with duplicated subject"""

    print()
    print("▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌")
    print("  THE XEROX PROJECT")
    print("  PHASE 4: INTERVIEW (Interactive Mode)")
    print("▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌")

    print("\nLoading duplicated subject...")
    model, tokenizer = load_model_and_tokenizer(weights_path=weights_path, model_name=model_name)

    print()
    print("="*60)
    print("  INTERVIEW SESSION ACTIVE")
    print("  The subject is ready. Type 'quit' to terminate.")
    print("="*60)
    print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n[SESSION TERMINATED]")
            break
        if not user_input:
            continue

        response = generate_response(model, tokenizer, user_input)
        print(f"\nSubject: {response}\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THE XEROX PROJECT - Phase 4: Interview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive interview
  python protocol/interview.py \\
      --weights subjects/001-elena-martinez/weights \\
      --interactive

  # Comparison mode (base vs duplicated)
  python protocol/interview.py \\
      --weights subjects/001-elena-martinez/weights \\
      --compare
        """
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to subject weights (LoRA adapter)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Base model")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive interview")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison mode (base vs duplicated)")

    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.weights, args.model)
    elif args.compare:
        run_comparison(args.weights, args.model)
    else:
        # Default: comparison mode
        run_comparison(args.weights, args.model)
