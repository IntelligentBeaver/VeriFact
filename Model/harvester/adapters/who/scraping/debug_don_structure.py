"""
Debug script to inspect HTML structure of a disease outbreak news article.
Fetches a sample article and prints its structure.
"""
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (WHO-scraper/1.0)"}

# Sample disease outbreak news URL
url = "https://www.who.int/emergencies/disease-outbreak-news/item/2025-DON591"

print(f"Fetching: {url}")
resp = requests.get(url, headers=HEADERS, timeout=15)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

# Find the article container
article = soup.select_one("article.sf-detail-body-wrapper")
if not article:
    print("❌ Article container not found!")
    exit(1)

print("\n✅ Found article container")
print(f"Article classes: {article.get('class')}")

# Check for don-revamp class
if 'don-revamp' in article.get('class', []):
    print("✅ Has 'don-revamp' class")

# Inspect direct children
print("\n📋 Direct children of article:")
for i, child in enumerate(article.children):
    if isinstance(child, str):
        text = child.strip()
        if text:
            print(f"  {i}: TEXT: {text[:60]}")
    elif hasattr(child, 'name'):
        classes = ' '.join(child.get('class', []))
        id_attr = child.get('id', '')
        text_preview = child.get_text(strip=True)[:60]
        print(f"  {i}: <{child.name} class='{classes}' id='{id_attr}'> {text_preview}")

# Look for heading tags
print("\n📍 Heading tags (h1-h6):")
for heading in article.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
    print(f"  <{heading.name}> {heading.get_text(strip=True)[:60]}")

# Look for divs with strong tags
print("\n💪 Divs with <strong> tags:")
for div in article.find_all('div'):
    strong = div.find('strong', recursive=False)  # Direct child only
    if strong:
        text = strong.get_text(strip=True)
        print(f"  <strong> {text[:60]}")

# Look for paragraphs
paragraphs = article.find_all('p')
print(f"\n📝 Paragraphs found: {len(paragraphs)}")
if paragraphs:
    print("  First 3 paragraphs:")
    for p in paragraphs[:3]:
        text = p.get_text(strip=True)
        print(f"    - {text[:80]}")

# Look for tables
tables = article.find_all('table')
print(f"\n📊 Tables found: {len(tables)}")

# Look for images
images = article.find_all('img')
print(f"\n🖼️  Images found: {len(images)}")

# Look for lists
lists = article.find_all(['ul', 'ol'])
print(f"\n📋 Lists found: {len(lists)}")

print("\n" + "="*60)
print("Raw HTML of first 2000 characters of article:")
print("="*60)
print(str(article)[:2000])
