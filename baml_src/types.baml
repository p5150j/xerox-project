// Type definitions for LoRA training data pipeline

class AdvisorDomain {
  name string @description("Advisor's full name")
  primary_domain string @description("Main area of expertise, e.g. 'burnout recovery'")
  expertise_areas string[] @description("3-5 specific topics they cover")
  communication_style string @description("How they communicate, e.g. 'warm and empathetic'")
  target_audience string @description("Who they typically help")
  key_themes string[] @description("Recurring concepts in their advice")
}

class TrainingExample {
  instruction string @description("Realistic user question or situation")
  response string @description("Advisor's response in their authentic voice")
}
