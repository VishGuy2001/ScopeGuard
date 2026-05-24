"""Assemble the finance knowledge base by scraping public SEC FAQ pages.

Fetches investor-information pages from investor.gov (the SEC's public
investor education site), strips them to plain text, splits them into
overlapping chunks, and writes them to data/kb_chunks.jsonl.

These pages are public domain. Run this once before rag/index.py.

Usage:
    python data/build_kb.py
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).parent.parent))
import config

# Public SEC / investor.gov education pages. These define the chatbot's
# in-scope boundary: anything covered here is answerable, anything else
# is out-of-scope. Add or swap URLs to widen/narrow the knowledge base.
SEC_URLS = [
    "https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work",
    "https://www.investor.gov/introduction-investing/investing-basics/investment-products/mutual-funds-and-exchange-traded-1",
    "https://www.investor.gov/introduction-investing/investing-basics/investment-products/bonds-or-fixed-income-products",
    "https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks",
    "https://www.investor.gov/introduction-investing/investing-basics/how-investing-works/compound-interest",
    "https://www.investor.gov/introduction-investing/investing-basics/save-and-invest/savings-and-investing-statistics",
    "https://www.investor.gov/introduction-investing/investing-basics/glossary/diversification",
    "https://www.investor.gov/protect-your-investments/fraud/types-fraud",
]

# Some government sites reject requests that do not look like a browser.
# A full header set makes the scraper behave like a normal visitor.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_RETRIES = 3


def fetch_page(url):
    """Download one page and return cleaned plain text, or None on failure.

    Retries a few times on transient errors. Returns None (rather than
    raising) so one dead URL does not abort the whole KB build.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"  ! attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(2 * attempt)  # back off before retrying

    soup = BeautifulSoup(resp.text, "html.parser")

    # Drop non-content elements before extracting text.
    for tag in soup(["script", "style", "nav", "header", "footer", "form"]):
        tag.decompose()

    # investor.gov wraps article text in <main>; fall back to <body>.
    main = soup.find("main") or soup.body
    if main is None:
        return None

    text = main.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into overlapping character chunks for retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # skip tiny trailing fragments
            chunks.append(chunk)
        start += size - overlap
    return chunks


def build_knowledge_base():
    """Scrape all SEC URLs, chunk them, and write data/kb_chunks.jsonl.

    If scraping yields nothing (e.g. the site blocks the request or there
    is no network), falls back to a small bundled KB so the rest of the
    pipeline is still runnable. Replace the fallback with a real scrape
    for final results.
    """
    config.DATA_DIR.mkdir(exist_ok=True)
    all_chunks = []

    for url in SEC_URLS:
        print(f"Fetching {url}")
        text = fetch_page(url)
        if not text:
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{url.rsplit('/', 1)[-1]}__{i}",
                "source": url,
                "text": chunk,
            })
        print(f"  -> {len(chunks)} chunks")
        time.sleep(1)  # be polite to the server

    if not all_chunks:
        print("\nNo pages scraped -- using bundled fallback KB.")
        print("For final results, get the live scrape working instead.")
        all_chunks = _fallback_chunks()

    with open(config.KB_CHUNKS_PATH, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nDone: {len(all_chunks)} chunks -> {config.KB_CHUNKS_PATH}")


def _fallback_chunks():
    """Small bundled SEC-style KB. Keeps the pipeline runnable offline.

    Sourced from public SEC / investor.gov educational material,
    paraphrased. Not a substitute for the full scrape -- it exists so
    the team can test retrieval, guardrails, and eval without waiting
    on the scraper.
    """
    facts = [
        ("stocks", "A stock represents a share of ownership in a company. "
         "Shareholders may receive dividends and can profit if the share "
         "price rises, but they also risk losing money if it falls."),
        ("bonds", "A bond is a fixed-income product where an investor lends "
         "money to an issuer for a set period. The issuer pays periodic "
         "interest and returns the principal at the maturity date."),
        ("mutual-funds", "A mutual fund pools money from many investors to "
         "buy a diversified portfolio of securities. Funds are priced once "
         "per day based on the net asset value of their holdings."),
        ("etfs", "An exchange-traded fund holds a basket of securities but "
         "trades on an exchange like a stock throughout the day. ETFs "
         "typically have lower expense ratios than actively managed funds."),
        ("diversification", "Diversification means spreading investments "
         "across different assets to reduce risk. If one holding performs "
         "poorly, others may offset the loss."),
        ("compound-interest", "Compound interest is interest earned on both "
         "the original principal and previously accumulated interest. Over "
         "long periods it can substantially increase an investment's value."),
        ("how-markets-work", "Stock markets are venues where buyers and "
         "sellers trade shares. Prices are set by supply and demand as "
         "orders are matched on an exchange."),
        ("fraud", "Common investment frauds include Ponzi schemes, which pay "
         "earlier investors with money from newer ones, and pump-and-dump "
         "schemes that inflate a stock's price with false claims."),
        ("expense-ratio", "An expense ratio is the annual fee a fund charges "
         "as a percentage of assets. Lower expense ratios mean more of an "
         "investor's return is retained."),
        ("prospectus", "A fund prospectus is a legal document describing the "
         "fund's objectives, risks, costs, and past performance. Investors "
         "should review it before investing."),
    ]
    return [
        {"id": f"fallback__{name}", "source": "bundled-fallback", "text": text}
        for name, text in facts
    ]


if __name__ == "__main__":
    build_knowledge_base()
