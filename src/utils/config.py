import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Loop & Safety Limits
MAX_TRIAGE_FALLBACK_ATTEMPTS = int(os.getenv("MAX_TRIAGE_FALLBACK_ATTEMPTS", 1))
MAX_PARALLEL_COMPETITORS = int(os.getenv("MAX_PARALLEL_COMPETITORS", 3))

# Recency Filters
RECENCY_CUTOFF_DAYS = int(os.getenv("RECENCY_CUTOFF_DAYS", 365))

# Shared extraction LLM (gpt-4o-mini, structured output)
extraction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
