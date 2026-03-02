from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import chromadb
import requests
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


@dataclass
class RAGDoc:
    title: str
    url: str
    text: str


class ModalExamplesRAG:
    """Vector DB RAG over Modal docs using ChromaDB.

    The vector collection is treated as the single source of truth for answers.
    """

    def __init__(
        self,
        cache_root: str = "/root/rag",
        collection_name: str = "modal_docs",
        max_pages: int = 80,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.persist_dir = self.cache_root / "chroma"
        self.meta_path = self.cache_root / "meta.json"
        self.collection_name = collection_name
        self.max_pages = max_pages

        self.seed_urls = [
            "https://modal.com/docs/examples",
            "https://modal.com/docs/guide",
            "https://modal.com/docs",
        ]

        self.client: chromadb.PersistentClient | None = None
        self.collection = None

    def load(self, max_cache_age_s: int = 86400) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        if self._is_fresh(max_cache_age_s=max_cache_age_s) and self.collection.count() > 0:
            return

        docs = self._crawl_docs()
        self._rebuild_collection(docs)
        self._write_meta(len(docs))

    def retrieve(self, query: str, k: int = 4) -> list[dict[str, str]]:
        if not self.collection:
            return []

        normalized_query = self.normalize_text(query)
        if not normalized_query:
            return []

        results = self.collection.query(query_texts=[normalized_query], n_results=k)
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        out: list[dict[str, str]] = []
        for text, meta in zip(docs, metadatas):
            out.append(
                {
                    "title": (meta or {}).get("title", "Modal docs"),
                    "url": (meta or {}).get("url", "https://modal.com/docs"),
                    "text": (text or "")[:520],
                }
            )
        return out

    def _crawl_docs(self) -> list[RAGDoc]:
        urls = self._discover_links()[: self.max_pages]
        docs: list[RAGDoc] = []
        for url in urls:
            parsed = self._fetch_doc(url)
            if not parsed:
                continue
            title, blocks = parsed
            for block in self._chunk_text(blocks, max_chars=800):
                normalized_block = self.normalize_text(block)
                if not normalized_block:
                    continue
                docs.append(RAGDoc(title=title, url=url, text=normalized_block))
        return docs

    def _discover_links(self) -> list[str]:
        collected: list[str] = []
        seen: set[str] = set()

        for seed in self.seed_urls:
            for link in self._discover_links_for_seed(seed):
                if link in seen:
                    continue
                seen.add(link)
                collected.append(link)

        return collected or ["https://modal.com/docs"]

    def _discover_links_for_seed(self, seed: str) -> list[str]:
        try:
            resp = requests.get(seed, timeout=20)
            resp.raise_for_status()
        except Exception:
            return [seed]

        soup = BeautifulSoup(resp.text, "html.parser")
        base = f"{urlparse(seed).scheme}://{urlparse(seed).netloc}"

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
            if not parsed.path.startswith("/docs"):
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
        title = (soup.title.string or "Modal docs").strip() if soup.title else "Modal docs"
        main = soup.find("main") or soup.find("article") or soup.body
        if not main:
            return None

        blocks: list[str] = []
        for node in main.find_all(["h1", "h2", "h3", "p", "li", "code"]):
            text = " ".join(node.get_text(" ", strip=True).split())
            if len(text) < 16:
                continue
            blocks.append(text)

        if not blocks:
            return None
        return title, blocks

    def normalize_text(self, text: str) -> str:
        """Shared normalization for both documents and user queries."""
        out = (text or "").strip().lower()
        out = re.sub(r"\s+", " ", out)
        return out

    def _chunk_text(self, blocks: list[str], max_chars: int = 800) -> list[str]:
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0

        for block in blocks:
            size = len(block)
            if cur and cur_len + size + 1 > max_chars:
                chunks.append(" ".join(cur))
                cur = [block]
                cur_len = size
            else:
                cur.append(block)
                cur_len += size + 1

        if cur:
            chunks.append(" ".join(cur))
        return chunks

    def _rebuild_collection(self, docs: list[RAGDoc]) -> None:
        if not self.collection:
            return

        existing = self.collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

        if not docs:
            return

        self.collection.add(
            ids=[f"doc_{i}" for i in range(len(docs))],
            documents=[d.text for d in docs],
            metadatas=[{"title": d.title, "url": d.url} for d in docs],
        )

    def _is_fresh(self, max_cache_age_s: int) -> bool:
        if not self.meta_path.exists():
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        ts = float(meta.get("ts", 0))
        return (time.time() - ts) <= max_cache_age_s

    def _write_meta(self, doc_count: int) -> None:
        self.meta_path.write_text(
            json.dumps({"ts": time.time(), "doc_count": doc_count}, ensure_ascii=True),
            encoding="utf-8",
        )
