// LLM client configuration for LoRA training data pipeline

client<llm> ClaudeSonnet {
  provider anthropic
  options {
    model "claude-sonnet-4-20250514"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 8192
  }
}

// Retry policy for handling transient failures
retry_policy Exponential {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}
