# Quick Start (No Theater Edition)

Skip the corporate dystopia aesthetic? Here's the technical rundown.

## What This Does

Fine-tunes LoRA adapters on Mistral 7B using synthetic training data generated via Claude's web search. Creates domain-specialized models that actually know things, not just roleplay knowing them.

## Setup

```bash
pip install anthropic python-dotenv baml-py torch transformers peft accelerate bitsandbytes

cp .env.example .env
# Edit .env with your Anthropic API key (for training corpus generation)
```

## The Pipeline

### 1. Generate Training Data

```bash
python3 protocol/synthesize.py \
    --source subjects/001-elena-martinez/source.txt \
    --output subjects/001-elena-martinez/training_data \
    --examples 250
```

**What happens:** Reads system prompt → Claude researches the domain via web search → Generates Q&A pairs in the persona's voice → Saves as JSONL.

**Outputs:**
- `*_research.md` - Raw research gathered
- `*_raw.json` - Training examples as JSON
- `*_mistral.jsonl` - Formatted for training

### 2. Train LoRA Adapter

```bash
python3 protocol/duplicate.py \
    --data subjects/001-elena-martinez/training_data/*_mistral.jsonl \
    --output subjects/001-elena-martinez/weights
```

**What happens:** Loads Mistral 7B with 4-bit quantization → Applies LoRA to attention layers → Trains for 3 epochs → Saves adapter.

**Config defaults:**
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- LoRA rank: 16, alpha: 32
- Target: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Epochs: 3, batch: 4, lr: 2e-4

**Override:**
```bash
python3 protocol/duplicate.py \
    --data ... \
    --output ... \
    --epochs 5 \
    --batch-size 2 \
    --lora-r 32
```

### 3. Test

**Compare base vs fine-tuned:**
```bash
python3 protocol/interview.py \
    --weights subjects/001-elena-martinez/weights \
    --compare
```

**Interactive chat:**
```bash
python3 protocol/interview.py \
    --weights subjects/001-elena-martinez/weights \
    --interactive
```

**A/B Comparison:**
```bash
python3 protocol/evaluate.py \
    --weights subjects/001-elena-martinez/weights
```

## Create Your Own

1. Create `subjects/002-your-expert/source.txt` with a system prompt
2. Run synthesize → duplicate → interview
3. Iterate on example count until quality is acceptable

## Hardware

- **RTX 4090 (24GB):** 7B models, ~2-5 min training
- **A100 (40GB+):** 70B models possible
- **CPU:** Inference only, slow

## File Structure

```
protocol/
  synthesize.py    # Data generation
  duplicate.py     # LoRA training
  interview.py     # Test/chat
  evaluate.py      # A/B comparison

subjects/
  001-elena-martinez/
    source.txt       # System prompt
    training_data/   # Generated data
    weights/         # LoRA adapter
```

## Troubleshooting

**CUDA OOM:** Reduce `--batch-size` to 2 or 1

**Slow downloads:** Set `HF_TOKEN` in .env

**BAML errors:** Run `baml-cli generate` after editing .baml files

---

That's it. The README has the full theatrical experience if you want it.
