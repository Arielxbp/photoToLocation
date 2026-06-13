import gzip
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

PLONKIT_BASE = "https://www.plonkit.net"
GUIDE_INDEX_URL = f"{PLONKIT_BASE}/guide"

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
FETCH_TIMEOUT = 60
FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_retryable(status_code: int) -> bool:
    return status_code in (429, 500, 502, 503, 504)


def _urlopen(url: str, timeout: int = FETCH_TIMEOUT) -> urllib.request._UrlopenRet:
    """Open a URL with browser-like headers and handle gzip encoding."""
    req = urllib.request.Request(url, headers=FETCH_HEADERS)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp


def _read_html(resp) -> str:
    """Read response body, handling gzip Content-Encoding."""
    data = resp.read()
    encoding = resp.headers.get("Content-Encoding", "")
    if "gzip" in encoding.lower():
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="replace")


def _extract_script_json(html: str, tag_id: str = "__PRELOADED_DATA__") -> Optional[dict]:
    """Extract and parse a <script id=... type=application/json> from HTML using regex
    (avoids BeautifulSoup for speed and to eliminate the bs4 dependency)."""
    pattern = (
        r'<script\s+[^>]*id\s*=\s*["\']' + re.escape(tag_id)
        + r'["\'][^>]*type\s*=\s*["\']application/json["\'][^>]*>'
        r'(.*?)</script>'
    )
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(1))


class _HTTPError(Exception):
    """Wrapper around urllib HTTP errors with status code access."""
    def __init__(self, code: int, msg: str):
        self.code = code
        super().__init__(msg)


def fetch_country_list() -> list[dict]:
    """Fetch the full country list with slugs from the Plonkit guide index page."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = _urlopen(GUIDE_INDEX_URL)
            if _is_retryable(resp.status):
                raise _HTTPError(resp.status, f"HTTP {resp.status} for {GUIDE_INDEX_URL}")
            html = _read_html(resp)
            data = _extract_script_json(html)
            if data is None:
                raise ValueError("Could not find __PRELOADED_DATA__ script tag on guide index")
            return data.get("data", [])
        except urllib.error.HTTPError as e:
            if _is_retryable(e.code) and attempt < MAX_RETRIES - 1:
                delay = RETRY_BACKOFF ** (attempt + 1)
                print(f"  [RETRY] Guide index ({e.code}, attempt {attempt + 1}/{MAX_RETRIES}), "
                      f"retrying in {delay:.0f}s")
                time.sleep(delay)
                continue
            raise _HTTPError(e.code, str(e))
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BACKOFF ** (attempt + 1)
                print(f"  [RETRY] Guide index failed (attempt {attempt + 1}/{MAX_RETRIES}), "
                      f"retrying in {delay:.0f}s: {e}")
                time.sleep(delay)
    raise last_exc


def fetch_country_guide(slug: str) -> dict:
    """Fetch the structured guide data for a single country."""
    url = f"{PLONKIT_BASE}/{slug}"
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = _urlopen(url)
            if resp.status == 404:
                raise _HTTPError(404, f"404 not found: {url} (guide may not be published yet)")
            if _is_retryable(resp.status):
                raise _HTTPError(resp.status, f"HTTP {resp.status} for {url}")
            html = _read_html(resp)
            data = _extract_script_json(html)
            if data is None:
                raise ValueError(f"Could not find __PRELOADED_DATA__ for /{slug}")
            public = data.get("data", {}).get("public", {})
            if not public:
                raise ValueError(f"No public data found for /{slug}")
            return public
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise _HTTPError(404, f"404 not found: {url}")
            if _is_retryable(e.code) and attempt < MAX_RETRIES - 1:
                delay = RETRY_BACKOFF ** (attempt + 1)
                print(f"  [RETRY] /{slug} ({e.code}, attempt {attempt + 1}/{MAX_RETRIES}), "
                      f"retrying in {delay:.0f}s")
                time.sleep(delay)
                continue
            raise _HTTPError(e.code, str(e))
        except _HTTPError as e:
            if e.code == 404:
                raise
            if _is_retryable(e.code) and attempt < MAX_RETRIES - 1:
                delay = RETRY_BACKOFF ** (attempt + 1)
                print(f"  [RETRY] /{slug} ({e.code}, attempt {attempt + 1}/{MAX_RETRIES}), "
                      f"retrying in {delay:.0f}s")
                time.sleep(delay)
                continue
            raise
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BACKOFF ** (attempt + 1)
                print(f"  [RETRY] /{slug} (attempt {attempt + 1}/{MAX_RETRIES}), "
                      f"retrying in {delay:.0f}s: {e}")
                time.sleep(delay)
                continue
            raise last_exc
    raise last_exc


TAG_TO_CLUE_CATEGORY = {
    "pole": "poles",
    "bollard": "bollards",
    "guardrail": "guardrails",
    "sign": "signage",
    "car": "car_meta",
}


def extract_clues_from_steps(steps: list[dict]) -> dict[str, list[str]]:
    """Parse guide steps and extract tagged clues into categorized lists."""
    clues: dict[str, list[str]] = {}

    for step in steps:
        if step.get("kind") != "tip":
            continue
        items = step.get("items", [])
        for item in items:
            tags = item.get("tags", [])
            if not tags:
                continue
            text_parts = item.get("data", {}).get("text", [])
            full_text = " ".join(text_parts) if isinstance(text_parts, list) else str(text_parts)
            full_text_clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", full_text)
            full_text_clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", full_text_clean)
            full_text_clean = re.sub(r"<[^>]+>", "", full_text_clean)
            full_text_clean = " ".join(full_text_clean.split())

            for tag in tags:
                category = TAG_TO_CLUE_CATEGORY.get(tag, tag)
                if category not in clues:
                    clues[category] = []
                if full_text_clean and full_text_clean not in clues[category]:
                    clues[category].append(full_text_clean)

    return clues


def parse_driving_side(text_parts: list[str]) -> Optional[str]:
    full = " ".join(text_parts).lower()
    if "left hand side" in full or "left-hand side" in full or "drives on the left" in full:
        return "left"
    if "right hand side" in full or "right-hand side" in full or "drives on the right" in full:
        return "right"
    return None


def extract_meta_clues(steps: list[dict]) -> dict[str, str | list[str]]:
    """Extract driving side, script/language, and other meta-level clues from guide steps."""
    meta: dict[str, str | list[str]] = {}

    for step in steps:
        if step.get("kind") != "tip":
            continue
        for item in step.get("items", []):
            text_parts = item.get("data", {}).get("text", [])
            if not text_parts:
                continue

            driving = parse_driving_side(text_parts)
            if driving:
                meta["driving_side"] = driving

            full = " ".join(text_parts).lower()
            for script in ["kanji", "hiragana", "katakana", "hangul", "cyrillic",
                           "thai", "arabic", "devanagari", "latin", "chinese"]:
                if script in full:
                    if "script" not in meta:
                        meta["script"] = []
                    if script not in meta["script"]:
                        meta["script"].append(script)

            for soil in ["red soil", "black soil", "sandy soil", "dry soil"]:
                if soil in full:
                    meta["soil"] = soil.split()[0]

    return meta


def build_clue_kb(cache_dir: Optional[str | Path] = None) -> dict[str, dict]:
    """
    Build the full clue knowledge base from all Plonkit country guides.
    Returns dict keyed by ISO country code (e.g. 'JP', 'GH').

    If cache_dir is provided, caches each country's raw JSON and
    skips re-fetching if already cached.
    """
    countries = fetch_country_list()
    kb: dict[str, dict] = {}
    cache_dir = Path(cache_dir) if cache_dir else None

    missing_pages = []
    failed_pages = []

    for entry in countries:
        code = entry.get("code")
        slug = entry.get("slug")
        title = entry.get("title", "")
        if not code or not slug:
            continue
        if code.startswith("XX-"):
            continue

        cached_file = cache_dir / f"{code}.json" if cache_dir else None
        if cached_file and cached_file.exists():
            guide = json.loads(cached_file.read_text())
        else:
            try:
                guide = fetch_country_guide(slug)
            except _HTTPError as e:
                if e.code == 404:
                    print(f"  [SKIP] {code} ({title}): guide not published yet (404)")
                    missing_pages.append(code)
                else:
                    print(f"  [FAIL] {code} ({title}): HTTP {e.code}: {e}")
                    failed_pages.append(code)
                continue
            except Exception as e:
                print(f"  [FAIL] {code} ({title}): {e}")
                failed_pages.append(code)
                continue
            if cached_file:
                cached_file.parent.mkdir(parents=True, exist_ok=True)
                cached_file.write_text(json.dumps(guide, indent=2))

        steps = guide.get("steps", [])
        clues = extract_clues_from_steps(steps)
        meta = extract_meta_clues(steps)

        entry_data: dict = {"name": title, "slug": slug}
        entry_data.update(meta)
        entry_data.update(clues)
        kb[code] = entry_data

    print(f"\nKB build summary:")
    print(f"  Collected: {len(kb)} countries")
    if missing_pages:
        print(f"  Missing (404): {len(missing_pages)} — {', '.join(sorted(missing_pages))}")
    if failed_pages:
        print(f"  Failed (error): {len(failed_pages)} — {', '.join(sorted(failed_pages))}")

    return kb


def load_clue_kb(cache_dir: str | Path) -> dict[str, dict]:
    """Load the clue KB from cached JSON files."""
    kb: dict[str, dict] = {}
    for f in Path(cache_dir).glob("*.json"):
        code = f.stem
        guide = json.loads(f.read_text())
        steps = guide.get("steps", [])
        clues = extract_clues_from_steps(steps)
        meta = extract_meta_clues(steps)
        entry_data: dict = {
            "name": guide.get("title", ""),
            "slug": guide.get("slug", ""),
        }
        entry_data.update(meta)
        entry_data.update(clues)
        kb[code] = entry_data
    return kb


def get_country_clue_summary(kb: dict[str, dict], code: str) -> str:
    """Return a human-readable summary of clues for a country."""
    entry = kb.get(code, {})
    if not entry:
        return f"No data for {code}"

    lines = [f"{entry.get('name', code)} ({code})"]
    for key in ["driving_side", "script", "soil", "car_meta", "poles", "bollards",
                "guardrails", "road_lines", "plates", "signage", "architecture", "landscape"]:
        val = entry.get(key)
        if val:
            if isinstance(val, list):
                lines.append(f"  {key}: {', '.join(str(v)[:120] for v in val[:3])}")
            else:
                lines.append(f"  {key}: {val}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    cache = sys.argv[1] if len(sys.argv) > 1 else "data/plonkit_cache"
    print(f"Building Plonkit KB → {cache}")
    kb = build_clue_kb(cache_dir=cache)
    print(f"\nCollected {len(kb)} countries")
    for code in sorted(kb)[:5]:
        print(f"\n{get_country_clue_summary(kb, code)}")
