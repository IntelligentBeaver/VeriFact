import json
import os

import requests, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import time

HEADERS = {"User-Agent": "Mozilla/5.0 (WHO-scraper/1.0)"}
HEADLINES_DIR = "../../storage/who/headlines"
OUTPUT_DIR = "../../storage/who/news"

SAVE_EVERY = 5  # save every 5 articles

os.makedirs(OUTPUT_DIR, exist_ok=True)

from bs4 import BeautifulSoup, Tag, NavigableString
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag, NavigableString
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag, NavigableString
from urllib.parse import urljoin


def parse_article_sections(article_div, base_url):
    """
    Parse an article div into sections with heading, content as list of lines, bullets, tables, and images.
    """
    sections = []

    # Find heading tags (h2, h3, h4) as section markers
    heading_tags = article_div.find_all(["h2", "h3", "h4"])

    if not heading_tags:
        # fallback: treat entire content as one section
        lines = []
        for el in article_div.children:
            if isinstance(el, Tag):
                if el.name == "p":
                    text = el.get_text(" ", strip=True)
                    if text:
                        lines.append(text)
                elif el.name in ["ul", "ol"]:
                    items = [li.get_text(" ", strip=True) for li in el.find_all("li")]
                    if items:
                        lines.extend(items)
        sections.append({"heading": None, "content": lines if lines else None})
    else:
        for i, h in enumerate(heading_tags):
            heading = h.get_text(strip=True)
            content_lines = []
            bullets = []
            tables = []
            images = []

            # iterate siblings until next heading at same or higher level
            for sib in h.next_siblings:
                if isinstance(sib, Tag) and sib.name in ["h2", "h3", "h4"]:
                    break
                if isinstance(sib, Tag):
                    if sib.name == "p":
                        text = sib.get_text(" ", strip=True)
                        if text:
                            # split by internal line breaks if needed
                            content_lines.extend(text.splitlines())
                    elif sib.name in ["ul", "ol"]:
                        items = [li.get_text(" ", strip=True) for li in sib.find_all("li")]
                        if items:
                            bullets.extend(items)
                    elif sib.name == "table":
                        rows = []
                        for tr in sib.find_all("tr"):
                            cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                            if cols:
                                rows.append(cols)
                        if rows:
                            tables.append(rows)
                    elif sib.name == "figure" or sib.find("img"):
                        imgs = sib.find_all("img")
                        for im in imgs:
                            src = im.get("src") or im.get("data-src")
                            if src:
                                images.append(urljoin(base_url, src))
                    else:
                        text = sib.get_text(" ", strip=True)
                        if text:
                            content_lines.extend(text.splitlines())
                elif isinstance(sib, NavigableString):
                    text = sib.strip()
                    if text:
                        content_lines.append(text)

            # Build the section object
            section_obj = {
                "heading": heading,
                "content": content_lines if content_lines else None,
            }
            if bullets:
                section_obj["bullets"] = bullets
            if tables:
                section_obj["tables"] = tables
            if images:
                section_obj["images"] = images

            sections.append(section_obj)

    return sections

def load_existing_items(filepath):
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_urls_from_year_file(filepath, min_year=2020):
    with open(filepath, "r", encoding="utf-8") as f:
        items = json.load(f)

    urls = []
    for item in items:
        date = item.get("date")
        if not date:
            continue

        year = int(date[:4])
        if year < min_year:
            continue

        urls.append(item["url"])

    return urls

def parse_date_str(s):
    if not s:
        return None
    s = s.strip()
    fmts = ["%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y"]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except Exception:
            pass
    # try zfill day
    parts = s.split()
    if len(parts) >= 3:
        day = parts[0].zfill(2)
        try:
            return datetime.strptime(f"{day} {parts[1]} {parts[2]}", "%d %B %Y").date().isoformat()
        except Exception:
            pass
    return None

def extract_media_contacts(block):
    # `block` is a BeautifulSoup element containing the media contacts block
    contacts = []
    if not block:
        return contacts
    # naive approach: split by name headings (they appear as plain text blocks)
    # Better approach: find sequences of <p> or block-level elements.
    # Here we look for groupings: name (text), role (text), then contact lines
    text_lines = [t.get_text(" ", strip=True) for t in block.find_all(recursive=False)]
    # fallback: parse using regex block search
    # We'll attempt a conservative parse: find lines that look like emails, phones and names
    raw = block.get_text("\n", strip=True)
    # split into persons by double newlines or by occurrence of 'Spokesperson'/'Communication Officer'
    persons = re.split(r'\n{2,}', raw)
    for p in persons:
        p = p.strip()
        if not p:
            continue
        # find emails and phones
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', p)
        phones = re.findall(r'(\+?\d[\d\s\-\(\)]+)', p)
        # first line as name (approx)
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        name = lines[0] if lines else None
        role = lines[1] if len(lines) > 1 else None
        contacts.append({
            "name": name,
            "role": role,
            "emails": emails or None,
            "phones": phones or None,
            "raw": p
        })
    return contacts

def scrape_who_news_item(url):
    """
    Robust scraper for WHO news item pages.
    - Uses the <article class="sf-detail-body-wrapper"> (or the article tag) as the main content area.
    - Extracts title, date, type, location, reading_time (value after 'Reading time:'), lead, content_html, content_text,
      bullets, references (from content), corrigendum, media contacts, topics, images and meta.
    - Avoids capturing site navigation by limiting to the article container.
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    base = resp.url
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else None

    # Date
    date_el = soup.select_one("span.timestamp")
    date = parse_date_str(date_el.get_text(strip=True)) if date_el else None

    # Type: look near the date; use regex to avoid precedence issues
    item_type = None
    if date_el:
        parent = date_el.parent
        if parent:
            # find any nearby tag that mentions the type explicitly
            type_candidate = parent.find(
                lambda tag: tag.name in ("div", "p", "span", "strong")
                and re.search(r"\b(News release|Statement|News|Feature story)\b", tag.get_text(), re.I)
            )
            if type_candidate:
                item_type = type_candidate.get_text(strip=True)
    if not item_type:
        # fallback: any early text on the page that looks like a type
        header_near = soup.find(string=re.compile(r"\b(News release|Statement|News|Feature story)\b", re.I))
        if header_near:
            item_type = header_near.strip()

    # Location/dateliner: look for uppercase token immediately preceding article first paragraph
    location = None
    # prefer article first paragraph (we will find article below)
    #  Change this later if WHO changes
    article = soup.select_one("article.sf-detail-body-wrapper, article, div.sf-detail-body-wrapper")
    first_para = None
    if article:
        first_para = article.select_one("p")
    else:
        first_para = soup.select_one("div.main-content p, article p, .sf-content-block p")

    if first_para:
        sibs = list(first_para.previous_siblings)
        for s in reversed(sibs):
            txt = getattr(s, "get_text", lambda **k: str(s))().strip()
            if txt and txt.upper() == txt and 1 <= len(txt) < 80:
                location = txt
                break

    # Reading time: find the element that contains 'Reading time' and extract the value after colon
    reading_time = None
    rt_node = soup.find(string=re.compile(r"Reading time", re.I))
    if rt_node:
        # check the full line / parent text and extract after colon
        parent = rt_node.parent if hasattr(rt_node, "parent") else None
        text_source = parent.get_text(" ", strip=True) if parent else str(rt_node)
        m = re.search(r"Reading time[:\s]*[:\-–]?\s*(.+)$", text_source, re.I)
        if m:
            val = m.group(1).strip()
            # sometimes text includes extra trailing words; keep the first reasonable chunk (up to newline)
            val = val.split("\n", 1)[0].strip()
            # if the value is just the label (i.e., "Reading time:"), skip
            if val and not re.match(r"^reading time[:\s]*$", val, re.I):
                reading_time = val

    # Lead (first meaningful paragraph inside article)
    lead = None
    if first_para:
        lead = first_para.get_text(strip=True)

    # Main article container: prioritize the article element to avoid menus/nav
    # content_container = article or soup.select_one("div.sf-content-block, div.content, div.main-content, div.page__content, main")
    content_container = soup.select_one("article.sf-detail-body-wrapper div")
    sections = parse_article_sections(content_container, resp.url) if content_container else []

    if not content_container:
        content_container = soup  # fallback, but we prefer article

    # CONTENT: use inner HTML of the article container (not entire page)
    # decode_contents() returns the HTML inside the tag (no wrapper)
    content_html = content_container.decode_contents() if hasattr(content_container, "decode_contents") else None
    content_text = content_container.get_text("\n", strip=True) if content_container else None

    # Bulleted lists inside the article only
    bullets = []
    if content_container:
        for li in content_container.select("ul li"):
            bullets.append(li.get_text(" ", strip=True))

    # References / related links inside content (anchor tags inside content only)
    references = []
    if content_container:
        for a in content_container.select("a[href]"):
            href = a["href"]
            full = urljoin(base, href)
            text = a.get_text(" ", strip=True)
            references.append({"text": text, "url": full})

    # Corrigendum / footnotes (search in article)
    corrigendum = None
    if content_container:
        corr_el = content_container.find(string=re.compile(r"Corrigendum|corrigendum", re.I))
        if corr_el:
            corrigendum = corr_el.strip()

    # Media contacts: find a header that mentions "Media" and grab following siblings until next section header
    contacts = []
    mc_header = soup.find(lambda tag: tag.name in ("h2", "h3", "strong", "p")
                          and re.search(r"\bmedia\b", tag.get_text(), re.I))
    if mc_header:
        block = mc_header.find_next_sibling()
        nodes = []
        cur = block
        # collect until next h2/h3/hr or until we've collected a few nodes
        while cur and not (cur.name in ("h2", "h3") or cur.name == "hr"):
            nodes.append(cur)
            cur = cur.find_next_sibling()
        if nodes:
            bs = BeautifulSoup("<div>" + "".join(str(n) for n in nodes) + "</div>", "html.parser")
            contacts = extract_media_contacts(bs.div)

    # Topics/tags: prefer tags inside the article page, fallback to site-level tags
    topics = [t.get_text(strip=True) for t in soup.select("div.topic-tags a, .topics a, .sf-tags-list-item")] or None

    # Images: try og:image first, then data-image/style/src within article/content
    og_image = None
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        og_image = urljoin(base, og["content"])

    images = []
    # prefer images inside content_container/article
    if content_container:
        for d in content_container.select("div.background-image, .hero img, figure img, img"):
            di = d.get("data-image") if d.has_attr("data-image") else None
            if di:
                images.append(urljoin(base, di))
                continue
            if d.name == "img" and d.get("src"):
                images.append(urljoin(base, d.get("src")))
                continue
            style = d.get("style") or ""
            m = re.search(r'url\((?:&quot;|")?(.*?)(?:&quot;|")?\)', style)
            if m:
                images.append(urljoin(base, m.group(1)))
    # fall back to og:image if nothing found inside content
    if not images and og_image:
        images = [og_image]

    # Meta description: keep it only if it's not just a verbatim duplicate of the article text
    meta_desc = None
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        candidate = md["content"].strip()
        # if meta description is included entirely within content_text (or vice-versa), drop it to avoid duplication
        if content_text and (candidate in content_text or content_text in candidate):
            meta_desc = None
        else:
            meta_desc = candidate

    result = {
        "url": base,
        "title": title,
        "date": date,
        "topics": topics,
        "type": item_type,
        "location": location,
        "reading_time": reading_time,       # now only the value after label
        "lead": lead,
        "content_html": content_html,       # inner HTML of the article
        "content": sections,
        "bullets": bullets,
        "references": references,
        "corrigendum": corrigendum,
        "media_contacts": contacts,
        "images": images,
        "meta": {"description": meta_desc, "og_image": og_image},
    }
    return result

def main():
    for filename in sorted(os.listdir(HEADLINES_DIR)):
        if not filename.endswith(".json"):
            continue

        year = filename.replace(".json", "")
        input_path = os.path.join(HEADLINES_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{year}.json")

        print(f"\nProcessing year {year}...")

        urls = load_urls_from_year_file(input_path, min_year=2006)

        year_results = load_existing_items(output_path)
        existing_urls = {item["url"] for item in year_results if "url" in item}

        for i, url in enumerate(urls, 1):
            print(f"  [{i}/{len(urls)}] {url}")

            if url in existing_urls:
                print("⏭️ Skipped (already scraped)")
                continue

            try:
                data = scrape_who_news_item(url)
                if not data:
                    continue

                year_results.append(data)
                existing_urls.add(url)

                if len(year_results) % SAVE_EVERY == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(year_results, f, ensure_ascii=False, indent=2)
                    print("💾 Progress saved")

            except Exception as e:
                print(f"❌ Error: {e}")

            time.sleep(0.1)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(year_results, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(year_results)} articles for {year}")

if __name__ == "__main__":
    main()
