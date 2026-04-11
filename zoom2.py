"""
Cult Beauty → WooCommerce CSV Scraper (Multithreaded)
======================================================
Scrapes: category PLPs (paginated) → individual product pages
Output:  woocommerce_products.csv

Threading model:
  - PLP pagination runs sequentially (one category at a time)
  - Product page requests run in parallel (PRODUCT_THREADS workers)
  - A threading.Lock protects the shared seen_skus set and rows list
  - Each thread has its own random delay so requests are staggered

RUN:
  pip install requests beautifulsoup4 lxml
  python cultbeauty_scraper.py
"""

import csv
import json
import re
import time
import random
import logging
import threading
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL        = "https://www.cultbeauty.com"
OUTPUT          = Path("woocommerce_products.csv")
MAX_PAGES       = None   # None = all pages; set e.g. 3 to limit during testing
PRODUCT_THREADS = 5      # parallel product page workers (keep ≤ 8 to be polite)
DELAY_MIN       = 1.0    # per-thread delay between requests
DELAY_MAX       = 2.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

# ── Category URLs ─────────────────────────────────────────────────────────────
# (url, woocommerce_category_path)
# Only real PLPs — no hub pages, no brand pages
CATEGORY_URLS = [

    # ── SKIN CARE ─────────────────────────────────────────────────────────────
    ("https://www.cultbeauty.com/c/skin-care/moisturisers/",            "Skin Care > Face Moisturisers"),
    ("https://www.cultbeauty.com/c/skin-care/serums/",                  "Skin Care > Serums"),
    ("https://www.cultbeauty.com/c/skin-care/cleansers-toners/",        "Skin Care > Cleansers & Toners"),
    ("https://www.cultbeauty.com/c/skin-care/toners-mists/toner/",      "Skin Care > Toners & Mists"),
    ("https://www.cultbeauty.com/c/skin-care/lips/",                    "Skin Care > Lip Care"),
    ("https://www.cultbeauty.com/c/spf/shop-all/",                      "Skin Care > SPF"),

    # ── FRAGRANCE ─────────────────────────────────────────────────────────────
    ("https://www.cultbeauty.com/c/fragrance/shop-all/",                "Fragrance > Shop All"),
    ("https://www.cultbeauty.com/c/fragrance/perfumes/",                "Fragrance > Perfumes"),
    ("https://www.cultbeauty.com/c/fragrance/discovery-sets/",          "Fragrance > Discovery Sets"),
    ("https://www.cultbeauty.com/c/fragrance/aftershave/",              "Fragrance > Aftershave"),
    ("https://www.cultbeauty.com/c/fragrance/home-fragrance/",          "Fragrance > Home Fragrance"),
    ("https://www.cultbeauty.com/c/hair-care/hair-mists/",              "Fragrance > Hair Mists"),

    # ── MAKE UP ───────────────────────────────────────────────────────────────
    ("https://www.cultbeauty.com/c/make-up/complexion/",                "Make Up > Complexion"),
    ("https://www.cultbeauty.com/c/make-up/eyes/",                      "Make Up > Eyes"),
    ("https://www.cultbeauty.com/c/make-up/lips/",                      "Make Up > Lips"),
    ("https://www.cultbeauty.com/c/make-up/brows/",                     "Make Up > Brows"),
    ("https://www.cultbeauty.com/c/make-up/brushes/",                   "Make Up > Brushes & Tools"),

    # ── BODY & WELLBEING ──────────────────────────────────────────────────────
    ("https://www.cultbeauty.com/c/body-wellbeing/moisturisers/",       "Body & Wellbeing > Body Moisturisers"),
    ("https://www.cultbeauty.com/c/body-wellbeing/cleansers/",          "Body & Wellbeing > Body Cleansers"),
    ("https://www.cultbeauty.com/c/body-wellbeing/body-oil-shimmers/",  "Body & Wellbeing > Body Oils & Shimmers"),
    ("https://www.cultbeauty.com/c/body-wellbeing/scrubs-exfoliators/", "Body & Wellbeing > Scrubs & Exfoliators"),
    ("https://www.cultbeauty.com/c/body-wellbeing/personal-care/",      "Body & Wellbeing > Personal Care"),
    ("https://www.cultbeauty.com/c/body-wellbeing/vitamins-supplements/","Body & Wellbeing > Vitamins & Supplements"),
    ("https://www.cultbeauty.com/c/body-wellbeing/tanning-suncare/tanning/","Body & Wellbeing > Tanning & Sun Care"),
    ("https://www.cultbeauty.com/c/body-wellbeing/wellbeing/aromatherapy/","Body & Wellbeing > Aromatherapy"),
    ("https://www.cultbeauty.com/c/body-wellbeing/wellbeing/sleep/",    "Body & Wellbeing > Sleep"),
    ("https://www.cultbeauty.com/c/body-wellbeing/hands-feet/hand-cream/","Body & Wellbeing > Hands & Feet"),
    ("https://www.cultbeauty.com/c/body-wellbeing/wellbeing/female-wellness/","Body & Wellbeing > Female Wellness"),
    ("https://www.cultbeauty.com/c/body-wellbeing/tools-technology/",   "Body & Wellbeing > Tools & Technology"),
]

# ── WooCommerce CSV columns ───────────────────────────────────────────────────
WC_COLUMNS = [
    "ID", "Type", "SKU", "Name", "Published", "Is featured?",
    "Visibility in catalog", "Short description", "Description",
    "Date sale price starts", "Date sale price ends",
    "Tax status", "Tax class", "In stock?", "Stock",
    "Backorders allowed?", "Sold individually?",
    "Weight (kg)", "Length (cm)", "Width (cm)", "Height (cm)",
    "Allow customer reviews?", "Purchase note",
    "Sale price", "Regular price",
    "Categories", "Tags", "Shipping class",
    "Images",
    "Download limit", "Download expiry days",
    "Parent", "Grouped products", "Upsells", "Cross-sells",
    "External URL", "Button text", "Position",
    "Attribute 1 name", "Attribute 1 value(s)",
    "Attribute 1 visible", "Attribute 1 global",
    "Attribute 2 name", "Attribute 2 value(s)",
    "Attribute 2 visible", "Attribute 2 global",
    "Meta: _yoast_wpseo_title", "Meta: _yoast_wpseo_metadesc",
    "Brands",
]

# ── Thread-local session (each thread reuses its own connection pool) ─────────
_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _local.session = s
    return _local.session


# ── HTTP ──────────────────────────────────────────────────────────────────────
def get(url: str, retries: int = 3) -> BeautifulSoup | None:
    session = get_session()
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            r = session.get(url, timeout=25)
            r.raise_for_status()
            r.encoding = "utf-8"  # fix accented chars (Néroli, Nécessaire etc.)
            return BeautifulSoup(r.text, "lxml")
        except Exception as exc:
            log.warning(f"Attempt {attempt + 1} failed for {url}: {exc}")
            time.sleep(4 * (attempt + 1))
    log.error(f"Giving up on {url}")
    return None


# ── PLP helpers ───────────────────────────────────────────────────────────────
def parse_breadcrumbs(soup: BeautifulSoup) -> str:
    nav = soup.find("nav", {"aria-label": "Breadcrumbs"})
    if not nav:
        return ""
    items = nav.find_all("li")
    parts = [li.get_text(strip=True) for li in items[1:]]
    parts = [p for p in parts if p and p.lower() != "home"]
    return " > ".join(parts)


def get_product_links(soup: BeautifulSoup) -> list[str]:
    links = []
    for card in soup.select("product-card-wrapper"):
        a = card.select_one("a.product-item")
        if a and a.get("href"):
            links.append(urljoin(BASE_URL, a["href"]))
    return list(dict.fromkeys(links))


def get_next_page(soup: BeautifulSoup) -> str | None:
    a = soup.select_one("a.next-page-button[data-hasmore='true']")
    if not a or not a.get("href"):
        return None
    href = a["href"]
    return href if href.startswith("http") else urljoin(BASE_URL, href)


# ── Product page helpers ──────────────────────────────────────────────────────
def extract_default_variant(soup: BeautifulSoup) -> dict:
    for script in soup.find_all("script"):
        text = script.string or ""
        if "defaultVariant" not in text:
            continue
        m = re.search(
            r"const\s+defaultVariant\s*=\s*(\{.+?\});\s*(?:\n|window)",
            text, re.S,
        )
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return {}


def html_to_text(html_str: str) -> str:
    if not html_str:
        return ""
    return BeautifulSoup(html_str, "lxml").get_text(" ", strip=True)


def get_content_value(content_list: list, key: str) -> str:
    for item in content_list:
        if item.get("key") != key:
            continue
        val = item.get("value", {})
        tn  = val.get("__typename", "")
        if tn == "ProductContentRichContentValue":
            parts = val.get("richContentValue", {}).get("content", [])
            return " ".join(p.get("content", "") for p in parts)
        if tn == "ProductContentRichContentListValue":
            out = []
            for rc in val.get("richContentListValue", []):
                out += [p.get("content", "") for p in rc.get("content", [])]
            return " ".join(out)
        if tn == "ProductContentStringValue":
            return val.get("stringValue", "")
        if tn == "ProductContentStringListValue":
            return ", ".join(val.get("stringListValue", []))
    return ""


def extract_sku(url: str) -> str:
    m = re.search(r"/p/[^/]+/(\d+)/", url)
    return m.group(1) if m else ""


def extract_variants(variant_data: dict) -> tuple[str, str]:
    choices = variant_data.get("choices", [])
    if not choices:
        return "", ""
    has_colour = any(c.get("colour") for c in choices)
    attr_name = "Shade" if has_colour else "Size"
    values = [c.get("title") or c.get("key", "") for c in choices if c.get("title") or c.get("key")]
    return attr_name, "|".join(values)


# ── Product page parser (called from worker threads) ──────────────────────────
def parse_product_page(url: str, fallback_category: str) -> dict | None:
    soup = get(url)
    if not soup:
        return None

    sku = extract_sku(url)

    h1 = soup.find("h1", id="product-title")
    title = h1.get_text(" ", strip=True) if h1 else ""
    if not title:
        t = soup.find("title")
        title = t.get_text(strip=True).split("|")[0].strip() if t else ""

    brand_a = soup.select_one("a.override-pdp-brand")
    brand = brand_a.get_text(strip=True) if brand_a else ""

    bc = parse_breadcrumbs(soup)
    category = bc if bc else fallback_category

    # Variant JSON extracted FIRST — used by price, images, content, variants
    variant_data = extract_default_variant(soup)

    # Price — 3 fallback levels
    price = ""
    if variant_data:
        try:
            price = str(
                variant_data.get("price", {})
                            .get("price", {})
                            .get("amount", "")
            )
        except Exception:
            pass
    if not price:
        price_el = soup.find(id="product-price")
        if price_el:
            price = re.sub(r"[€£$,\s]", "", price_el.get_text(strip=True))
    if not price:
        for sel in [".price span", ".product-price span", "span.price"]:
            p_el = soup.select_one(sel)
            if p_el:
                candidate = re.sub(r"[€£$,\s]", "", p_el.get_text(strip=True))
                if re.match(r"^\d+\.?\d*$", candidate):
                    price = candidate
                    break

    # Images
    images = []
    for img in variant_data.get("images", []):
        orig = img.get("original", "")
        if orig:
            images.append(
                orig if orig.startswith("http")
                else f"https://static.thcdn.com/productimg/original/{Path(orig).name}"
            )
    if not images:
        for img_tag in soup.select(".carouselImages img, .product-images img"):
            src = img_tag.get("src") or img_tag.get("data-src", "")
            if "productimg" in src and src not in images:
                images.append(src)

    # Content
    content     = variant_data.get("content", [])
    why_choose  = html_to_text(get_content_value(content, "whyChoose"))
    synopsis    = html_to_text(get_content_value(content, "synopsis"))
    ingredients = html_to_text(get_content_value(content, "ingredients"))
    directions  = html_to_text(get_content_value(content, "directions"))
    compliance  = html_to_text(get_content_value(content, "complianceWarnings"))

    short_desc = synopsis or why_choose

    parts = []
    if why_choose:  parts.append(f"Why It's Cult:\n{why_choose}")
    if synopsis:    parts.append(synopsis)
    if ingredients: parts.append(f"Key Ingredients:\n{ingredients}")
    if directions:  parts.append(f"How to Use:\n{directions}")
    if compliance:  parts.append(f"Important Information:\n{compliance}")
    full_desc = "\n\n".join(parts)

    # Stock
    in_stock = "1"
    avail = soup.find(id="availabilityMessage")
    if avail and "out of stock" in avail.get_text(strip=True).lower():
        in_stock = "0"
    if variant_data and not variant_data.get("inStock", True):
        in_stock = "0"

    attr_name, attr_val = extract_variants(variant_data)

    tags_set = set()
    for seg in category.split(" > "):
        seg = seg.strip()
        if seg and seg not in ("Shop All",):
            tags_set.add(seg)
    if brand:
        tags_set.add(brand)
    tags = ", ".join(sorted(tags_set))

    return {
        "sku":               sku,
        "name":              title,
        "brand":             brand,
        "price":             price,
        "category":          category,
        "tags":              tags,
        "short_description": short_desc,
        "description":       full_desc,
        "images":            "|".join(images[:8]),
        "in_stock":          in_stock,
        "attr_name":         attr_name,
        "attr_val":          attr_val,
    }


# ── WC row builder ────────────────────────────────────────────────────────────
def build_wc_row(p: dict) -> dict:
    row = {col: "" for col in WC_COLUMNS}
    row["Type"]                        = "simple"
    row["SKU"]                         = p["sku"]
    row["Name"]                        = p["name"]
    row["Published"]                   = "1"
    row["Is featured?"]                = "0"
    row["Visibility in catalog"]       = "visible"
    row["Short description"]           = p["short_description"]
    row["Description"]                 = p["description"]
    row["Tax status"]                  = "taxable"
    row["In stock?"]                   = p["in_stock"]
    row["Backorders allowed?"]         = "0"
    row["Sold individually?"]          = "0"
    row["Allow customer reviews?"]     = "1"
    row["Regular price"]               = p["price"]
    row["Categories"]                  = p["category"]
    row["Tags"]                        = p["tags"]
    row["Images"]                      = p["images"]
    row["Brands"]                      = p["brand"]
    row["Attribute 1 name"]            = "Brand"
    row["Attribute 1 value(s)"]        = p["brand"]
    row["Attribute 1 visible"]         = "1"
    row["Attribute 1 global"]          = "1"
    if p.get("attr_name") and p.get("attr_val"):
        row["Attribute 2 name"]        = p["attr_name"]
        row["Attribute 2 value(s)"]    = p["attr_val"]
        row["Attribute 2 visible"]     = "1"
        row["Attribute 2 global"]      = "1"
    row["Meta: _yoast_wpseo_title"]    = p["name"][:60]
    row["Meta: _yoast_wpseo_metadesc"] = p["short_description"][:160]
    return row


# ── Worker function (runs in thread pool) ─────────────────────────────────────
def scrape_product(args: tuple) -> dict | None:
    """Called by thread pool. Returns parsed product dict or None."""
    prod_url, fallback_category = args
    product = parse_product_page(prod_url, fallback_category)
    if product:
        log.info(
            f"  ✓ [{product['sku']}] {product['name'][:50]}"
            f" | {product['price']}€ | {product['category']}"
        )
    else:
        log.warning(f"  ✗ Failed: {prod_url}")
    return product


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    seen_skus: set[str] = set()
    seen_lock = threading.Lock()
    rows: list[dict] = []

    log.info(
        f"Scraping {len(CATEGORY_URLS)} categories | "
        f"threads={PRODUCT_THREADS} | MAX_PAGES={MAX_PAGES or 'unlimited'}"
    )

    for cat_url, cat_label in CATEGORY_URLS:
        log.info(f"\n{'='*60}\nCATEGORY: {cat_label}\n{cat_url}")
        page_url  = cat_url
        page_num  = 0
        cat_count = 0

        while page_url:
            page_num += 1
            if MAX_PAGES and page_num > MAX_PAGES:
                log.info(f"MAX_PAGES={MAX_PAGES} reached, next category")
                break

            log.info(f"  [Page {page_num}] {page_url}")
            # PLP page fetched on the main thread
            soup = get(page_url)
            if not soup:
                break

            bc       = parse_breadcrumbs(soup)
            fallback = bc if bc else cat_label
            links    = get_product_links(soup)
            log.info(f"  Found {len(links)} product links")

            if not links:
                log.warning("  No products — end of list or unexpected page type")
                break

            # Filter out already-seen SKUs before dispatching to thread pool
            new_links = []
            for prod_url in links:
                sku = extract_sku(prod_url) or prod_url
                with seen_lock:
                    if sku not in seen_skus:
                        seen_skus.add(sku)
                        new_links.append((prod_url, fallback))

            log.info(f"  Dispatching {len(new_links)} new products to {PRODUCT_THREADS} threads")

            # Scrape product pages in parallel
            with ThreadPoolExecutor(
                max_workers=PRODUCT_THREADS,
                thread_name_prefix="Worker"
            ) as executor:
                futures = {executor.submit(scrape_product, args): args for args in new_links}
                for future in as_completed(futures):
                    try:
                        product = future.result()
                        if product:
                            rows.append(build_wc_row(product))
                            cat_count += 1
                    except Exception as exc:
                        log.error(f"  Thread error: {exc}")

            page_url = get_next_page(soup)

        log.info(f"  → {cat_count} new products from {cat_label}")

    # Write CSV — done on main thread after all scraping complete
    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=WC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"\n{'='*60}")
    log.info(f"✅ DONE: {len(rows)} products saved to {OUTPUT}")
    log.info(f"   Unique SKUs: {len(seen_skus)}")


if __name__ == "__main__":
    main()