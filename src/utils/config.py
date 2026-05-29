import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Loop & Safety Limits
MAX_TRIAGE_FALLBACK_ATTEMPTS = int(os.getenv("MAX_TRIAGE_FALLBACK_ATTEMPTS", 1))
MAX_PARALLEL_COMPETITORS = int(os.getenv("MAX_PARALLEL_COMPETITORS", 3))
MAX_REMEDIATION_ATTEMPTS = int(os.getenv("MAX_REMEDIATION_ATTEMPTS", 2))
MAX_HITL_ITERATIONS = int(os.getenv("MAX_HITL_ITERATIONS", 2))

# Persistence
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "checkpoints.db")

# Context Management (ARCH-01)
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", 2048))
MAX_HISTORY_CHARS = MAX_HISTORY_TOKENS * 4  # heuristic: 1 token ~= 4 chars

# Recency Filters
RECENCY_CUTOFF_DAYS = int(os.getenv("RECENCY_CUTOFF_DAYS", 365))
STALE_SIGNAL_THRESHOLD_MONTHS = int(os.getenv("STALE_SIGNAL_THRESHOLD_MONTHS", 18))

# Shared extraction LLM (gpt-4o-mini, structured output)
extraction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Synthesis LLM: gpt-4o primary with gpt-4o-mini fallback
_primary_synthesis_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_fallback_synthesis_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
synthesis_llm = _primary_synthesis_llm.with_fallbacks([_fallback_synthesis_llm])
