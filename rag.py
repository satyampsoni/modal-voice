from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")


@dataclass
class RAGChunk:
    title: str
    url: str
    text: str
    tokens: list[str]


class ModalExamplesRAG:
    """Lightweight lexical RAG over Modal examples docs."""

    def __init__(
        self,
        seed_url: str = "https://modal.com/docs/examples",
        cache_path: str = "/root/rag/examples_index.json",
        max_pages: int = 60,
    ) -> None:
        self.seed_url = seed_url
        self.seed_urls = [
            "https://modal.com/docs/examples",
            "https://modal.com/docs/guide",
            "https://modal.com/docs",
        ]
        self.cache_path = Path(cache_path)
        self.max_pages = max_pages

        self.chunks: list[RAGChunk] = []
        self.idf: dict[str, float] = {}

    def load(self, max_cache_age_s: int = 86400) -> None:
        if self._load_cache(max_cache_age_s=max_cache_age_s):
            return
        self._build_index()
        self._save_cache()

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, str]]:
        if not self.chunks:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores: list[tuple[float, int]] = []
        for i, chunk in enumerate(self.chunks):
            if not chunk.tokens:
                continue

            term_hits = 0.0
            token_set = set(chunk.tokens)
            for qt in q_tokens:
                if qt in token_set:
                    term_hits += self.idf.get(qt, 1.0)

            if term_hits <= 0:
                continue

            # Slightly favor concise chunks for faster prompt processing.
            brevity_penalty = max(1.0, len(chunk.text) / 600.0)
            score = term_hits / brevity_penalty
            scores.append((score, i))

        scores.sort(reverse=True)

        results: list[dict[str, str]] = []
        for _, idx in scores[:k]:
            chunk = self.chunks[idx]
            results.append(
                {
                    "title": chunk.title,
                    "url": chunk.url,
                    "text": chunk.text[:420],
                }
            )
        return results

    def _build_index(self) -> None:
        links = self._discover_links()
        pages = links[: self.max_pages]

        chunks: list[RAGChunk] = []
        for url in pages:
            doc = self._fetch_doc(url)
            if not doc:
                continue
            title, paragraphs = doc
            for block in self._chunk_paragraphs(paragraphs):
                tokens = self._tokenize(block)
                if not tokens:
                    continue
                chunks.append(RAGChunk(title=title, url=url, text=block, tokens=tokens))

        self.chunks = chunks
        self._compute_idf()

    def _discover_links(self) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        for seed in self.seed_urls:
            discovered = self._discover_links_for_seed(seed)
            for u in discovered:
                if u in seen:
                    continue
                seen.add(u)
                urls.append(u)

        return urls or [self.seed_url]

    def _discover_links_for_seed(self, seed: str) -> list[str]:
        try:
            resp = requests.get(seed, timeout=20)
            resp.raise_for_status()
        except Exception:
            return [seed]

        soup = BeautifulSoup(resp.text, "html.parser")
        base = f"{urlparse(seed).scheme}://{urlparse(seed).netloc}"
        docs_prefixes = ("/docs/examples", "/docs/guide", "/docs")

        urls = [seed]
        seen = {seed}
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#"):
                continue
            absolute = urljoin(base, href)
            parsed = urlparse(absolute)
            if parsed.netloc != urlparse(seed).netloc:
                continue
            if not parsed.path.startswith(docs_prefixes):
                continue
            cleaned = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if cleaned in seen:
                continue
            seen.add(cleaned)
            urls.append(cleaned)
        return urls

    def _fetch_doc(self, url: str) -> tuple[str, list[str]] | None:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
        except Exception:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        title = (soup.title.string or "Modal Example").strip() if soup.title else "Modal Example"

        main = soup.find("main") or soup.find("article") or soup.body
        if not main:
            return None

        paragraphs: list[str] = []
        for p in main.find_all(["p", "li", "code", "h2", "h3"]):
            text = " ".join(p.get_text(" ", strip=True).split())
            if len(text) < 12:
                continue
            paragraphs.append(text)

        if not paragraphs:
            return None
        return title, paragraphs

    def _chunk_paragraphs(self, paragraphs: list[str], max_chars: int = 520) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        cur_len = 0

        for p in paragraphs:
            plen = len(p)
            if cur_len + plen + 1 > max_chars and current:
                chunks.append(" ".join(current))
                current = [p]
                cur_len = plen
            else:
                current.append(p)
                cur_len += plen + 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _compute_idf(self) -> None:
        df: dict[str, int] = {}
        n = max(1, len(self.chunks))

        for chunk in self.chunks:
            for token in set(chunk.tokens):
                df[token] = df.get(token, 0) + 1

        self.idf = {t: (1.0 + (n / (1 + freq))) for t, freq in df.items()}

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "chunks": [
                {
                    "title": c.title,
                    "url": c.url,
                    "text": c.text,
                    "tokens": c.tokens,
                }
                for c in self.chunks
            ],
        }
        self.cache_path.write_text(json.dumps(payload), encoding="utf-8")

    def _load_cache(self, max_cache_age_s: int) -> bool:
        if not self.cache_path.exists():
            return False

        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        ts = float(payload.get("ts", 0))
        if (time.time() - ts) > max_cache_age_s:
            return False

        raw_chunks = payload.get("chunks", [])
        chunks: list[RAGChunk] = []
        for item in raw_chunks:
            try:
                chunks.append(
                    RAGChunk(
                        title=item["title"],
                        url=item["url"],
                        text=item["text"],
                        tokens=list(item.get("tokens", [])),
                    )
                )
            except Exception:
                continue

        if not chunks:
            return False

        self.chunks = chunks
        self._compute_idf()
        return True
