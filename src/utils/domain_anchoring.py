from urllib.parse import urlparse

from .models import IntelligenceState


def domain_anchoring_node(state: IntelligenceState) -> dict:
    """
    Strips the subpath, query string, and fragment so the agent researches
    the company homepage, not a specific article or landing page.
    Normalises a leading "www." prefix.

    Design note: we keep the host as-is rather than taking only the last two
    dot-parts, so ccTLDs (company.co.uk) and intentional subdomains survive.

    Input examples → output:
      "https://expressanalytics.com/blog/article?q=1#top" → "expressanalytics.com"
      "www.stripe.com/pricing"                            → "stripe.com"
      "linear.app"                                        → "linear.app"
    """
    url = state.get("target_company_url", "")
    if not url:
        return {}

    # Handle URLs without a scheme (urlparse treats them as paths otherwise)
    if "://" not in url:
        netloc = url.split("/")[0]
    else:
        parsed = urlparse(url)
        netloc = parsed.netloc

    # Normalise www.
    if netloc.startswith("www."):
        netloc = netloc[4:]

    print(f"--- Domain anchored: {state['target_company_url']} -> {netloc} ---")
    return {"target_company_url": netloc}
