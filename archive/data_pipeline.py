#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()
"""
LoRA Training Data Pipeline for Advisor Personas
Gathers domain-specific data and synthesizes training examples.
"""

import json
import os
from pathlib import Path
from typing import Optional
import anthropic

from baml_client.sync_client import b
from baml_client.types import AdvisorDomain, TrainingExample

# ============================================================================
# Domain Extraction
# ============================================================================

def extract_domain(system_prompt: str) -> AdvisorDomain:
    """Extract domain information from advisor system prompt using BAML"""
    return b.ExtractDomain(system_prompt=system_prompt)

# ============================================================================
# Web Research (using Claude's web_search)
# ============================================================================

def research_domain(client: anthropic.Anthropic, domain: AdvisorDomain) -> str:
    """Gather web research for the advisor's domain"""

    search_prompt = f'''Research the following domain to gather expert knowledge:

DOMAIN: {domain.primary_domain}
EXPERTISE AREAS: {', '.join(domain.expertise_areas)}
KEY THEMES: {', '.join(domain.key_themes)}

Search for:
1. Expert advice and best practices in this domain
2. Common questions people ask about these topics
3. Research-backed strategies and frameworks
4. Real-world case studies or examples
5. Common misconceptions to address

For each finding, note:
- The source
- Key insights
- Practical advice given
- Questions being answered

Compile comprehensive research that could help an advisor in this domain.'''

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 10
        }],
        messages=[{"role": "user", "content": search_prompt}]
    )

    # Extract all text content
    research = ""
    for block in response.content:
        if hasattr(block, 'text'):
            research += block.text + "\n"

    return research

# ============================================================================
# Training Data Synthesis
# ============================================================================

def synthesize_training_data(
    domain: AdvisorDomain,
    research: str,
    system_prompt: str,
    num_examples: int = 100
) -> list[TrainingExample]:
    """Generate training examples using BAML structured outputs"""

    # Generate in batches to avoid token limits
    batch_size = 25
    all_examples = []

    for i in range(0, num_examples, batch_size):
        current_batch = min(batch_size, num_examples - i)
        print(f"  Generating batch {i//batch_size + 1} ({current_batch} examples)...")

        batch = b.SynthesizeTrainingData(
            name=domain.name,
            primary_domain=domain.primary_domain,
            expertise_areas=', '.join(domain.expertise_areas),
            communication_style=domain.communication_style,
            target_audience=domain.target_audience,
            research=research[:10000],
            system_prompt=system_prompt[:2000],
            num_examples=current_batch
        )
        all_examples.extend(batch)
        print(f"    Got {len(batch)} examples")

    return all_examples

# ============================================================================
# Output Formatting
# ============================================================================

def format_for_training(examples: list[TrainingExample], model_format: str = "mistral") -> list[str]:
    """Format examples for specific model training format"""
    formatted = []

    for ex in examples:
        if model_format == "mistral":
            # Mistral instruction format
            text = f"<s>[INST] {ex.instruction} [/INST] {ex.response}</s>"
        elif model_format == "llama":
            # Llama 3 format
            text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{ex.instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ex.response}<|eot_id|>"
        else:
            # Simple format
            text = f"User: {ex.instruction}\nAssistant: {ex.response}"

        formatted.append(text)

    return formatted

def save_training_data(examples: list[TrainingExample], output_dir: Path, advisor_name: str):
    """Save training data in multiple formats"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw JSON
    raw_path = output_dir / f"{advisor_name}_raw.json"
    with open(raw_path, 'w') as f:
        json.dump([ex.model_dump() for ex in examples], f, indent=2)
    print(f"  Saved raw JSON: {raw_path}")

    # Mistral format JSONL
    mistral_path = output_dir / f"{advisor_name}_mistral.jsonl"
    formatted = format_for_training(examples, "mistral")
    with open(mistral_path, 'w') as f:
        for text in formatted:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"  Saved Mistral JSONL: {mistral_path}")

    return mistral_path

# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
    system_prompt: str,
    output_dir: str = "./training_data",
    num_examples: int = 500,
    api_key: Optional[str] = None
):
    """Run the full data gathering and synthesis pipeline"""

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY required")

    client = anthropic.Anthropic(api_key=api_key)
    output_path = Path(output_dir)

    print("="*60)
    print("LoRA Training Data Pipeline")
    print("="*60)

    # Step 1: Extract domain
    print("\n[1/4] Extracting domain from system prompt...")
    domain = extract_domain(system_prompt)
    print(f"  Name: {domain.name}")
    print(f"  Domain: {domain.primary_domain}")
    print(f"  Expertise: {', '.join(domain.expertise_areas)}")

    # Step 2: Research
    print("\n[2/4] Gathering web research...")
    research = research_domain(client, domain)
    print(f"  Gathered {len(research)} characters of research")

    # Save research for reference
    research_path = output_path / f"{domain.name.lower().replace(' ', '_')}_research.md"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(research_path, 'w') as f:
        f.write(f"# Research for {domain.name}\n\n{research}")
    print(f"  Saved research: {research_path}")

    # Step 3: Synthesize training data
    print(f"\n[3/4] Synthesizing {num_examples} training examples...")
    examples = synthesize_training_data(domain, research, system_prompt, num_examples)
    print(f"  Generated {len(examples)} examples")

    # Step 4: Save
    print("\n[4/4] Saving training data...")
    advisor_slug = domain.name.lower().replace(' ', '_').replace('.', '')
    training_path = save_training_data(examples, output_path, advisor_slug)

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Training data: {training_path}")
    print(f"Examples: {len(examples)}")
    print(f"\nNext: Run training with:")
    print(f"  python train_lora.py --data {training_path}")

    return training_path

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LoRA training data for an advisor")
    parser.add_argument("--prompt-file", type=str, help="Path to file containing advisor system prompt")
    parser.add_argument("--prompt", type=str, help="Advisor system prompt (inline)")
    parser.add_argument("--output", type=str, default="./training_data", help="Output directory")
    parser.add_argument("--examples", type=int, default=500, help="Number of training examples")
    parser.add_argument("--api-key", type=str, help="Anthropic API key")

    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file) as f:
            system_prompt = f.read()
    elif args.prompt:
        system_prompt = args.prompt
    else:
        print("Error: Either --prompt-file or --prompt required")
        exit(1)

    run_pipeline(
        system_prompt=system_prompt,
        output_dir=args.output,
        num_examples=args.examples,
        api_key=args.api_key
    )
