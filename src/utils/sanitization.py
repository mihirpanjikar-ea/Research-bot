import re
from urllib.parse import urlparse, urlunparse

from .models import IntelligenceState

# Characters / sequences that could break prompt structure or inject instructions
_NEWLINE_RE = re.compile(r"[\r\n\x00-\x08\x0b\x0c\x0e-\x1f]")

# Prompt-injection trigger phrases (case-insensitive) that have no place in a URL
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|above|all)\s+instructions?"
    r"|system\s*prompt"
    r"|you\s+are\s+now"
    r"|act\s+as\s+"
    r"|do\s+not\s+follow)",
    re.IGNORECASE,
)

_ALLOWED_SCHEMES = {"http", "https", ""}


def sanitization_node(state: IntelligenceState) -> dict:
    """
    Validates and escapes the raw target_company_url before it touches any
    prompt or Exa call, closing the prompt-injection surface.

    Checks performed (in order):
    1. Reject empty input.
    2. Normalise scheme: prepend https:// if missing.
    3. Enforce http / https schemes only.
    4. Strip control characters and newlines from every URL component.
    5. Reject URLs whose decoded path/query contain prompt-injection phrases.
    6. Rebuild a clean URL (scheme + netloc + path, no query/fragment).

    Raises ValueError on rejection so LangGraph surfaces the error clearly
    rather than silently passing a dangerous string downstream.
    """
    raw = state.get("target_company_url", "").strip()

    if not raw:
        raise ValueError("target_company_url is empty.")

    # Add scheme if missing so urlparse can extract netloc correctly
    if "://" not in raw:
        raw = "https://" + raw

    parsed = urlparse(raw)

    # Enforce allowed schemes
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Scheme '{parsed.scheme}' is not allowed. Only http/https accepted."
        )

    # Strip control characters from every URL component
    netloc = _NEWLINE_RE.sub("", parsed.netloc)
    path = _NEWLINE_RE.sub("", parsed.path)
    query = _NEWLINE_RE.sub("", parsed.query)

    # Reject prompt-injection patterns found anywhere in the URL
    combined = netloc + path + query
    if _INJECTION_PATTERNS.search(combined):
        raise ValueError(
            f"URL contains a prompt-injection pattern and was rejected: {raw!r}"
        )

    # Rebuild: keep scheme + netloc + path; drop query and fragment
    sanitized = urlunparse((parsed.scheme, netloc, path, "", "", ""))

    if sanitized != raw:
        print(f"--- Sanitized URL: {raw!r} -> {sanitized!r} ---")

    return {"target_company_url": sanitized}
