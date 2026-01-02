# save as scrape_agri_schemes.py
import requests
from bs4 import BeautifulSoup
import time
import csv
import urllib.parse

BASE_URL = "https://agriwelfare.gov.in"
MAJOR_SCHEMES_PATH = "/en/Major"   # page used in this example

HEADERS = {
    "User-Agent": "AgriSchemesScraper/1.0 (+your-email@example.com) - For research use only"
}

def fetch(url, max_tries=3, backoff=1.0):
    for attempt in range(max_tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.text
        except Exception as e:
            wait = backoff * (2 ** attempt)
            print(f"Fetch failed ({e}), retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {max_tries} tries")

def parse_major_schemes(html, base_url=BASE_URL):
    soup = BeautifulSoup(html, "html.parser")
    schemes = []

    # --- ADJUST SELECTORS based on the page structure ---
    # On the ministry "Major" page there may be a list/table of schemes.
    # We'll look for link elements that look like scheme entries.
    for a in soup.select("a"):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        # Heuristic: scheme links often contain known keywords or link to /en/..
        if not text:
            continue
        if "/en/" in href or "scheme" in text.lower() or "pm" in text.lower() or len(text) < 80:
            # Normalize href
            full_url = urllib.parse.urljoin(base_url, href)
            schemes.append({
                "name": text,
                "url": full_url
            })
    # dedupe by url
    seen = set()
    uniq = []
    for s in schemes:
        if s["url"] not in seen:
            seen.add(s["url"])
            uniq.append(s)
    return uniq

def fetch_scheme_description(url):
    try:
        html = fetch(url)
        soup = BeautifulSoup(html, "html.parser")
        # Try common places for a description: first paragraph under main content
        p = soup.select_one("main p, .content p, .entry-content p, #main p")
        if p:
            return p.get_text(strip=True)[:600]  # truncate to 600 chars
        # Fallback: meta description
        meta = soup.find("meta", {"name":"description"})
        if meta and meta.get("content"):
            return meta["content"].strip()[:600]
    except Exception as e:
        print("Failed to fetch scheme page:", e)
    return ""

def main():
    start_url = urllib.parse.urljoin(BASE_URL, MAJOR_SCHEMES_PATH)
    print("Fetching:", start_url)
    html = fetch(start_url)
    schemes = parse_major_schemes(html)
    print(f"Found {len(schemes)} candidate links (heuristic-filtered).")

    results = []
    for i, s in enumerate(schemes, start=1):
        print(f"{i:03d}. {s['name']}\n    {s['url']}")
        # polite pause
        time.sleep(1.5)
        desc = fetch_scheme_description(s['url'])
        results.append({
            "name": s['name'],
            "url": s['url'],
            "description": desc
        })
        # very small pause between pages
        time.sleep(1.0)

    # write CSV
    with open("schemes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name","url","description"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("Saved schemes.csv with", len(results), "rows.")

if __name__ == "__main__":
    main()
