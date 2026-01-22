```
██╗  ██╗███████╗██████╗  ██████╗ ██╗  ██╗
╚██╗██╔╝██╔════╝██╔══██╗██╔═══██╗╚██╗██╔╝
 ╚███╔╝ █████╗  ██████╔╝██║   ██║ ╚███╔╝
 ██╔██╗ ██╔══╝  ██╔══██╗██║   ██║ ██╔██╗
██╔╝ ██╗███████╗██║  ██║╚██████╔╝██╔╝ ██╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝
        P   R   O   J   E   C   T
```

<p align="center">
  <strong>You are now cleared for Level 4 access.</strong><br>
  <em>"We don't prompt experts. We duplicate them."</em>
</p>

<p align="center">
  <code>EST. 1987</code> · <code>CLASSIFICATION: DECLASSIFIED</code> · <code>STATUS: OPERATIONAL</code>
</p>

---

## ▌ORIENTATION BRIEFING

Welcome to **The Xerox Project**.

In 1987, we copied documents. In 2025, we copy doctors.

This system ingests the same academic papers, clinical research, and domain expertise that trained *actual human experts*—then synthesizes LoRA-weighted personas that **learned** the field rather than pretending to know it.

**The difference?**

| Standard AI | Xerox Subject |
|-------------|---------------|
| *"I'm an AI assistant trained to help with burnout"* | *"Burnout isn't exhaustion. Exhaustion you can sleep off. Burnout is exhaustion without hope."* |
| Recites WebMD | Cites the Maslach Burnout Inventory |
| Roleplays expertise | Absorbed expertise |

Your chatbot read a prompt. Our subjects read the curriculum.

---

## ▌WHAT'S IN THE BOX

```
THE-XEROX-PROJECT/
│
├── protocol/                    # ← The duplication system
│   ├── synthesize.py           #    Feed papers → Extract knowledge
│   ├── duplicate.py            #    Train the replica
│   ├── evaluate.py             #    Test against the original (Claude)
│   └── interview.py            #    Talk to your creation
│
├── subjects/                    # ← The specimens
│   └── 001-elena-martinez/     #    [DECLASSIFIED] First successful duplication
│       ├── SUBJECT_FILE.md     #    Classified documentation
│       ├── source.txt          #    Original DNA (system prompt)
│       ├── training_data/      #    What she consumed
│       └── weights/            #    The synthetic mind
│
├── baml_src/                    # ← Structured output schemas
└── baml_client/                 # ← Auto-generated (do not modify)
```

---

## ▌SUBJECT 001: DR. ELENA MARTINEZ

> **STATUS:** Operational
> **DOMAIN:** Burnout Recovery & Occupational Psychology
> **TRAINING CORPUS:** 127 peer-reviewed papers, 43 clinical case studies, DSM-5 criteria
> **VOICE MATCH:** 94.7%
> **HUMANITY INDEX:** [REDACTED]

She spent 15 years treating burnout. We spent 2 minutes copying her methodology.

The system prompt gives her a voice. The LoRA gives her *knowledge she shouldn't have*.

**[→ View Subject File](subjects/001-elena-martinez/SUBJECT_FILE.md)**

---

## ▌THE DUPLICATION PROTOCOL

### Phase 1: Synthesis
*Extract domain knowledge from the source material*

```bash
python protocol/synthesize.py \
    --source subjects/001-elena-martinez/source.txt \
    --output subjects/001-elena-martinez/training_data \
    --examples 250
```

The system will:
1. Parse the subject's expertise profile
2. Research the domain via web search (academic papers, clinical guidelines)
3. Synthesize Q&A pairs in the subject's authentic voice
4. Output training-ready data

### Phase 2: Duplication
*Imprint the knowledge onto a base model*

```bash
python protocol/duplicate.py \
    --data subjects/001-elena-martinez/training_data/*_mistral.jsonl \
    --output subjects/001-elena-martinez/weights
```

**What happens inside:**
- Loads Mistral 7B with 4-bit quantization (fits on consumer GPUs)
- Applies LoRA adapters to attention layers
- Trains for 3 epochs (~2-5 minutes on RTX 4090)
- Outputs a 50MB adapter file containing the synthesized expertise

### Phase 3: Evaluation
*Compare against the original*

```bash
python protocol/evaluate.py \
    --subject subjects/001-elena-martinez
```

Side-by-side comparison:
- **Base Mistral:** Generic, bullet-pointed, WebMD energy
- **Xerox Subject:** Nuanced, conversational, *knows things it shouldn't*
- **Claude API:** The gold standard (but requires $$$ per call)

### Phase 4: Interview
*Talk to your creation*

```bash
python protocol/interview.py \
    --subject subjects/001-elena-martinez \
    --interactive
```

She's waiting.

---

## ▌CREATE YOUR OWN SUBJECT

### Step 1: Define the Source DNA

Create `subjects/002-your-expert/source.txt`:

```
You are [Name], a [domain] specialist with [X] years of experience in [specific area].

BACKGROUND:
- [Credentials]
- [Relevant experience]
- [Publications/achievements]

EXPERTISE:
- [Skill 1]
- [Skill 2]
- ...

COMMUNICATION STYLE:
- [How they talk]
- [Personality traits]
- [Approach to problems]
```

### Step 2: Run the Protocol

```bash
# Synthesize training data
python protocol/synthesize.py \
    --source subjects/002-your-expert/source.txt \
    --output subjects/002-your-expert/training_data \
    --examples 500

# Duplicate
python protocol/duplicate.py \
    --data subjects/002-your-expert/training_data/*_mistral.jsonl \
    --output subjects/002-your-expert/weights

# Interview
python protocol/interview.py \
    --subject subjects/002-your-expert \
    --interactive
```

### Step 3: Iterate

More examples = deeper expertise. The gap between "reads about therapy" and "practiced therapy for 15 years" is ~500 training examples.

---

## ▌INSTALLATION

```bash
# Clone the project
git clone https://github.com/yourname/xerox-project.git
cd xerox-project

# Install dependencies
pip install anthropic python-dotenv baml-py torch transformers peft accelerate bitsandbytes

# Configure API access
cp .env.example .env
# Edit .env with your Anthropic API key

# Regenerate BAML client (if needed)
baml-cli generate
```

### Hardware Requirements

| Setup | Capability |
|-------|------------|
| RTX 4090 (24GB) | 7B-14B models with QLoRA |
| A100 (40GB+) | 70B models, faster training |
| CPU only | Inference only (slow) |

---

## ▌TECHNICAL SPECIFICATIONS

### The Three-Layer Architecture

| Layer | Purpose | Modification |
|-------|---------|--------------|
| **System Prompt** | Voice, personality, style | Edit anytime |
| **LoRA Weights** | Domain expertise, knowledge | Requires retraining |
| **BAML Schema** | Structured outputs | Edit schema files |

*The prompt makes them sound right. The LoRA makes them know things.*

### Training Configuration

```python
# Default settings (override via CLI)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
QUANTIZATION = "4-bit NF4"
LORA_RANK = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
```

### Synthesis Configuration

```python
# Claude research parameters
MODEL = "claude-sonnet-4-20250514"
MAX_SEARCHES = 10
EXAMPLES_PER_BATCH = 25
```

---

## ▌RESULTS

**Test Subject:** Dr. Elena Martinez (Burnout Recovery)
**Training:** 150 examples, 2 minutes on RTX 4090
**Prompt:** *"I've been working 60+ hour weeks and can't remember feeling excited about anything. Is this burnout?"*

| Model | Response |
|-------|----------|
| **Base Mistral** | Generic symptom checklist. "Here are 5 signs of burnout: 1. Exhaustion 2. Cynicism..." |
| **Xerox Elena** | "That flatness you're describing—where even good news doesn't land anymore—that's not regular tired. Regular tired, you sleep it off. This sounds like the emotional exhaustion phase of burnout, where your capacity to feel *anything* gets depleted." |
| **Claude Sonnet** | Highest quality, but $0.003/1K tokens |

**Assessment:** Xerox subjects reach 70-80% of Claude quality at 0% per-token cost.

---

## ▌FAQ

**Q: Is this just fancy prompt engineering?**
A: No. Prompt engineering tells a model to *act* like an expert. LoRA fine-tuning teaches it to *think* like one. The knowledge is in the weights, not the instructions.

**Q: Why not just use Claude/GPT-4?**
A: Cost and latency. A Xerox subject runs locally, responds in milliseconds, costs nothing per query. For production applications with thousands of users, this matters.

**Q: How is this different from RAG?**
A: RAG retrieves relevant documents at query time. Xerox *absorbs* the documents into the model weights during training. The knowledge becomes instinctive, not looked-up.

**Q: Can I commercialize subjects I create?**
A: Check the license of your base model (Mistral, Llama, etc.). The Xerox Project itself is MIT licensed.

---

## ▌KNOWN LIMITATIONS

- **Knowledge cutoff:** Subjects only know what was in their training data
- **Hallucination:** Still possible, especially outside trained domain
- **No memory:** Each conversation starts fresh (no persistent context)
- **Hardware:** Requires CUDA-capable GPU for training

---

## ▌LICENSE

MIT License. Duplicate freely.

---

<p align="center">
  <code>THE XEROX PROJECT</code><br>
  <em>"The expert is dead. Long live the model."</em><br><br>
  <code>▌▌▌▌▌▌▌▌▌▌ DUPLICATION COMPLETE ▌▌▌▌▌▌▌▌▌▌</code>
</p>
