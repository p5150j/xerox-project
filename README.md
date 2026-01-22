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

In 1987, we copied documents. In 2026, we copy doctors.

This system ingests the same academic papers, clinical research, and domain expertise that trained *actual human experts*—then synthesizes LoRA-weighted personas that **learned** the field rather than pretending to know it.

**The difference?**

| Standard AI | Xerox Subject |
|-------------|---------------|
| *"I'm an AI assistant trained to help with burnout"* | *"Burnout isn't exhaustion. Exhaustion you can sleep off. Burnout is exhaustion without hope."* |
| Recites WebMD | Cites the Maslach Burnout Inventory |
| Roleplays expertise | Absorbed expertise |

Your chatbot reads a prompt. Our subjects read the curriculum.

---

## ▌WHAT'S IN THE BOX

```
THE-XEROX-PROJECT/
│
├── protocol/                    # ← The duplication system
│   ├── synthesize.py           #    Feed papers → Extract knowledge
│   ├── duplicate.py            #    Train the replica
│   ├── evaluate.py             #    Test against the original
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

> **STATUS:** Operational<br>
> **DOMAIN:** Burnout Recovery & Occupational Psychology<br>
> **TRAINING CORPUS:** 127 peer-reviewed papers, 43 clinical case studies, DSM-5 criteria<br>
> **VOICE MATCH:** 94.7%<br>
> **HUMANITY INDEX:** [REDACTED]

She spent 15 years treating burnout. We spent 2 minutes cloning her methodology.

The system prompt gives her a voice. The LoRA gives her *knowledge she shouldn't have*.

**[→ View Subject File](subjects/001-elena-martinez/SUBJECT_FILE.md)**

---

## ▌ACQUIRED INTELLIGENCE

> *"We don't make this up. We steal it from people who know things."*

Every Xerox subject is built on **real research**. Not vibes. Not prompt tricks. Actual peer-reviewed papers, clinical frameworks, and domain expertise scraped from the sources that trained the humans we're copying.

**During synthesis, the system executes live web searches:**

```
→ "burnout recovery clinical guidelines"
→ "Maslach Burnout Inventory assessment criteria"
→ "workplace boundary setting research"
→ "occupational psychology energy management"
→ "identity reconstruction after career crisis"
```

**What comes back:**
- [DSM-5 occupational problem criteria](https://www.psychiatry.org/psychiatrists/practice/dsm) + [ICD-11 burnout definition](https://icd.who.int/browse11/l-m/en#/http://id.who.int/icd/entity/129180281)
- [Peer-reviewed intervention studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4911781/)
- [Clinical treatment protocols](https://www.ncbi.nlm.nih.gov/books/NBK279286/)
- [Expert frameworks (Maslach, Leiter, Schaufeli)](https://www.wilmarschaufeli.nl/publications/Schaufeli/311.pdf)
- [Real practitioner interviews](https://hbr.org/2019/12/burnout-is-about-your-workplace-not-your-people)

This isn't a chatbot pretending to have read the textbook. This is a model that *actually consumed the curriculum*.

---

## ▌THE DUPLICATION PROTOCOL

### Phase 1: Synthesis
*Extract domain knowledge from the source material*

```bash
python3 protocol/synthesize.py \
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
python3 protocol/duplicate.py \
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
python3 protocol/evaluate.py \
    --weights subjects/001-elena-martinez/weights
```

Side-by-side comparison:
- **Base Mistral:** Generic, bullet-pointed, WebMD energy
- **Xerox Subject:** Nuanced, conversational, *knows things it shouldn't*

### Phase 4: Interview
*Talk to your creation*

```bash
python3 protocol/interview.py \
    --weights subjects/001-elena-martinez/weights \
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
python3 protocol/synthesize.py \
    --source subjects/002-your-expert/source.txt \
    --output subjects/002-your-expert/training_data \
    --examples 500

# Duplicate
python3 protocol/duplicate.py \
    --data subjects/002-your-expert/training_data/*_mistral.jsonl \
    --output subjects/002-your-expert/weights

# Interview
python3 protocol/interview.py \
    --subject subjects/002-your-expert \
    --interactive
```

### Step 3: Iterate

> **More examples = deeper expertise.**
>
> The gap between *"reads about therapy"* and *"practiced therapy for 15 years"* is **~500 training examples**.

---

## ▌INSTALLATION

```bash
# Clone the project
git clone https://github.com/p5150j/xerox-project.git
cd xerox-project

# Install dependencies
pip install anthropic python-dotenv baml-py torch transformers peft accelerate bitsandbytes

# Configure API access
cp .env.example .env
# Edit .env with your Anthropic API key (for training corpus generation)

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

**Assessment:** The difference is clear. Base Mistral recites information. Xerox subjects *understand* the domain.

---

## ▌FAQ

**Q: Is this real or a joke repo?**
A: Fuck yeah it's real. Every subject is grounded in peer-reviewed literature, clinical frameworks, and domain-specific epistemology. Give it a spin and clone anyone you can think of.

**Q: Is this just fancy prompt engineering?**
A: No. Prompt engineering tells a model to *act* like an expert. LoRA fine-tuning teaches it to *think* like one. The knowledge is in the weights, not the instructions.

**Q: Why not just use Claude/GPT-4?**
A: Two reasons. First: this is a fully trained LoRA that you **own**. It runs on your hardware, responds in milliseconds, and costs exactly $0.00 per query. Forever. No API keys, no rate limits, no billing surprises. You trained it. It's yours. Second: **depth over breadth**. Claude and GPT-4 know a little about everything. A Xerox subject knows *everything* about one thing. You curate exactly what goes in—want an alien tax advisor who grew up in Uganda and trained at clown school? You can build that. Try getting that out of a general-purpose API.

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

*If this creates a race of robotic lizard people in illuminati robes, I am not responsible. If you build something amazing and buy a Lambo, maybe ping me and buy me some tacos.* 🌮

---

<p align="center">
  <code>THE XEROX PROJECT</code><br>
  <em>"The expert is dead. Long live the model."</em><br><br>
  <code>▌▌▌▌▌▌▌▌▌▌ DUPLICATION COMPLETE ▌▌▌▌▌▌▌▌▌▌</code>
</p>
