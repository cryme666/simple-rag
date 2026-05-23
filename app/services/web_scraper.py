import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com/bot)"
    )
}

TAGS_TO_REMOVE = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]


async def scrape_url(url: str, timeout: float = 30.0) -> dict:
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(str(url), headers=HEADERS)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    for tag in soup.find_all(TAGS_TO_REMOVE):
        tag.decompose()

    body = soup.find("body")
    if body:
        text = body.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)

    return {"title": title, "text": text}
