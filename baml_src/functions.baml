// BAML functions for LoRA training data pipeline

function ExtractDomain(system_prompt: string) -> AdvisorDomain {
  client ClaudeSonnet
  prompt #"
    Analyze this advisor system prompt and extract structured information.

    ADVISOR PROMPT:
    {{ system_prompt }}

    Extract the advisor's name, primary domain, expertise areas,
    communication style, target audience, and key themes.

    {{ ctx.output_format }}
  "#
}

function SynthesizeTrainingData(
  name: string,
  primary_domain: string,
  expertise_areas: string,
  communication_style: string,
  target_audience: string,
  research: string,
  system_prompt: string,
  num_examples: int
) -> TrainingExample[] {
  client ClaudeSonnet
  prompt #"
    You are generating training data for fine-tuning an AI advisor.

    ADVISOR PROFILE:
    Name: {{ name }}
    Domain: {{ primary_domain }}
    Expertise: {{ expertise_areas }}
    Style: {{ communication_style }}
    Audience: {{ target_audience }}

    RESEARCH GATHERED:
    {{ research }}

    ORIGINAL SYSTEM PROMPT (for voice/tone reference):
    {{ system_prompt }}

    Generate {{ num_examples }} diverse training examples. Each example should be:
    1. A realistic user question/situation (instruction)
    2. The advisor's response in their authentic voice (response)

    Requirements:
    - Vary question types: direct questions, situations needing advice, emotional venting, specific scenarios
    - Responses should match the advisor's communication style exactly
    - Include domain-specific knowledge from the research
    - Responses should be 2-4 paragraphs (substantial but not overwhelming)
    - Make it feel like a real conversation, not a lecture

    {{ ctx.output_format }}
  "#
}
