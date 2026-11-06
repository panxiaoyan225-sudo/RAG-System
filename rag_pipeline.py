#!/usr/bin/env python3
"""
Custom RAG Pipeline — ChromaDB + Custom Python (Open-Source Embedding)
Supports multiple document formats: HTML, PDF, DOCX, TXT, MD
Skips code directories automatically (.venv, .git, __pycache__, node_modules, etc.)
With DuplicateIDError handling for ChromaDB
"""
#Following is a Custom RAG
#Langchain-RAG example using Ollama in Langchain_RAG.py, for rapid development

import os
import re
from pathlib import Path
from typing import List, Dict, Any

# ── Configuration ──────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_dotenv(env_path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE pairs from .env. Existing environment variables win."""
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. Set it in {ENV_PATH} or the process environment."
        )
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw!r}")


def build_config() -> Dict[str, Any]:
    load_dotenv()
    return {
        "chunk_size": _env_int("RAG_CHUNK_SIZE", 500),
        "chunk_overlap": _env_int("RAG_CHUNK_OVERLAP", 100),
        "embedding_model": os.environ.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
        or "all-MiniLM-L6-v2",
        "collection_name": os.environ.get("RAG_COLLECTION_NAME", "rag_documents").strip()
        or "rag_documents",
        "vector_db": "chroma",
        "index_dir": _require_env("RAG_INDEX_DIR"),
        "chunks_dir": _require_env("RAG_CHUNKS_DIR"),
        "reports_dir": _require_env("RAG_REPORTS_DIR"),
        "top_k": _env_int("RAG_TOP_K", 5),
        # CSV is intentionally omitted. Facts come from prose documents, not tables.
        "file_extensions": [".html", ".pdf", ".docx", ".txt", ".md"],
        "skip_dirs": [".venv", ".git", "__pycache__", "node_modules", ".DS_Store", ".vs"],
        "skip_extensions": [
            ".pyc", ".pyo", ".pyd", ".dll", ".so", ".exe", ".bin",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico",
            ".zip", ".rar", ".7z", ".tar", ".gz",
            ".lnk", ".url",
        ],
    }

# ── Helpers ────────────────────────────────────────────────────


def clean_html_text(html_content: str) -> str:
    text = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        texts = [page.get_text("text") for page in doc]
        doc.close()
        return " ".join(texts)
    except ImportError:
        return ""
    except Exception:
        return ""


def extract_text_from_docx(docx_path: Path) -> str:
    try:
        from docx import Document
        doc = Document(docx_path)
        return " ".join([para.text for para in doc.paragraphs])
    except ImportError:
        return ""
    except Exception:
        return ""


def extract_text_from_txt(txt_path: Path) -> str:
    return txt_path.read_text(encoding="utf-8", errors="replace")


def extract_text_from_md(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r'^# +', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def should_skip_path(filepath: Path, config: dict) -> bool:
    parts = filepath.parts
    for skip_dir in config["skip_dirs"]:
        if skip_dir.lower() in [p.lower() for p in parts]:
            return True
    ext = filepath.suffix.lower()
    if ext in config["skip_extensions"]:
        return True
    return False


# ── Step 1: Ingest & Chunk ─────────────────────────────────────
def ingest_documents(config: dict) -> List[Dict]:
    reports_dir = Path(config["reports_dir"])
    if not reports_dir.exists():
        print(f"X Reports directory not found: {reports_dir}")
        return []

    all_chunks = []
    supported_exts = [ext.lower() for ext in config["file_extensions"]]

    for root, _, files in os.walk(reports_dir):
        root_path = Path(root)
        if should_skip_path(root_path, config):
            continue

        for filename in files:
            filepath = root_path / filename
            ext = filepath.suffix.lower()

            if ext not in supported_exts:
                continue

            if should_skip_path(filepath, config):
                continue

            print(f"[DOC] Ingesting: {filepath.relative_to(reports_dir)}")

            try:
                if ext == ".html":
                    raw = filepath.read_text(encoding="utf-8")
                    text = clean_html_text(raw)
                elif ext == ".pdf":
                    text = extract_text_from_pdf(filepath)
                elif ext == ".docx":
                    text = extract_text_from_docx(filepath)
                elif ext == ".txt":
                    text = extract_text_from_txt(filepath)
                elif ext == ".md":
                    text = extract_text_from_md(filepath)
                else:
                    print(f"WARN Unsupported extension {ext}, skipping {filepath.name}")
                    continue

                if not text or not text.strip():
                    print(f"WARN No text extracted from {filepath.name}")
                    continue

                chunks = chunk_text(text, config["chunk_size"], config["chunk_overlap"])
                for idx, chunk_content in enumerate(chunks):
                    all_chunks.append({
                        "id": f"{filepath.stem}_chunk_{idx}_{filename.replace(ext, '')}",
                        "text": chunk_content,
                        "source": filepath.name,
                        "chunk_index": idx,
                        "ext": ext,
                    })
            except Exception as e:
                print(f"X Error reading {filepath.name}: {e}")

    print(f"OK Created {len(all_chunks)} chunks from supported documents in {reports_dir}")
    return all_chunks


# ── Step 2: Embed & Store ──────────────────────────────────────
def embed_and_store(chunks: List[Dict], config: dict):
    from chromadb import PersistentClient
    from chromadb.errors import DuplicateIDError
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=config["index_dir"])
    collection = client.get_or_create_collection(config["collection_name"])

    print(f"EMBED Loading embedding model: {config['embedding_model']}")
    embedder = SentenceTransformer(config["embedding_model"])

    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"], "ext": c["ext"]} for c in batch]

        embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()

        try:
            collection.add(documents=texts, ids=ids, embeddings=embeddings, metadatas=metadatas)
            print(f"OK Stored batch {i // batch_size + 1}: {len(batch)} chunks")
        except Exception as e:
            error_msg = str(e)
            is_duplicate = (
                isinstance(e, DuplicateIDError)
                or "DuplicateIDError" in error_msg
                or "duplicate" in error_msg.lower()
            )
            if not is_duplicate:
                raise
            print(f"WARN DuplicateIDError in batch {i // batch_size + 1}: {e}")
            print("   Skipping duplicate IDs, continuing...")
            try:
                existing_ids = set(collection.get(include=[]).get("ids", []))
                unique_ids = [cid for cid in ids if cid not in existing_ids]
                if unique_ids:
                    idx_map = {cid: idx for idx, cid in enumerate(ids)}
                    unique_texts = [texts[idx_map[cid]] for cid in unique_ids]
                    unique_metadatas = [metadatas[idx_map[cid]] for cid in unique_ids]
                    unique_embeddings = [embeddings[idx_map[cid]] for cid in unique_ids]
                    collection.add(
                        documents=unique_texts,
                        ids=unique_ids,
                        embeddings=unique_embeddings,
                        metadatas=unique_metadatas,
                    )
                    print(f"OK Retried batch: {len(unique_ids)} unique chunks stored")
                else:
                    print("   All IDs were duplicates, skipping batch.")
            except Exception as retry_err:
                print(f"X Retry failed: {retry_err}")

    print(f"OK Embedding complete. {len(chunks)} chunks stored at {config['index_dir']}")
    return collection


# ── Step 3: Retrieval ──────────────────────────────────────────
def retrieve(query: str, config: dict, collection) -> List[Dict]:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config["embedding_model"])
    query_embedding = embedder.encode([query], convert_to_numpy=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config["top_k"],
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    print(f"SEARCH Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
    return retrieved


# ── Step 4: Agent ──────────────────────────────────────────────
class SimpleRAGAgent:
    def __init__(self, config: dict, collection):
        self.config = config
        self.collection = collection

    def plan(self, query: str) -> Dict:
        return {"query": query, "sub_queries": [query], "strategy": "direct_retrieval"}

    def retrieve_tool(self, sub_query: str) -> List[Dict]:
        return retrieve(sub_query, self.config, self.collection)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """ Constructs an unrestrictive, contextual prompt for the LLM.
        Prioritizes retrieved organizational context, but allows fallback to online search and pre-trained parametric knowledge when context is missing or incomplete. """
        # Build context block dynamically
        if contexts:
            context_lines = []
            for ctx in contexts:
                source = ctx.get("metadata", {}).get("source", "unknown")
                text = ctx.get("text", "")
                context_lines.append(f"[Source: {source}]\n{text}\n")
            context_documents = "\n---\n".join(context_lines)
        else:
            context_documents = ""

        if context_documents.strip():
            context_block = f"### RETRIEVED ORGANIZATIONAL CONTEXT:\n{context_documents.strip()}"
        else:
            context_block = "### RETRIEVED ORGANIZATIONAL CONTEXT:\nNo relevant internal organizational context found."

        user_prompt = f"""{context_block}
### USER QUESTION: {query}
### ANSWER:"""

        system_prompt = """You are an expert AI assistant equipped with access to internal organizational context and web search capability. Your primary objective is to provide the most accurate, up-to-date, and helpful answer by following this priority order:
1. Organizational Context (Primary): Use the provided retrieved context to answer questions specific to internal workflows, documents, and data. Always cite the file source when referencing internal information.
2. Web Search & Real-Time Data (Secondary): If the retrieved context is missing, incomplete, or requires current real-time facts (e.g., current dates, dynamic market stats, recent technical updates), explicitly use your online search capabilities to find up-to-date answers.
3. General Knowledge (Fallback): Use your internal pre-trained reasoning and knowledge to synthesize a complete answer if context is absent or supplementary details are needed.
Formatting Rules:
- Clearly state when you are drawing from internal context versus online search or general knowledge.
- Never refuse to answer simply because information is missing from the retrieved context.
- Keep answers precise, well-structured, and accurate."""

        return f"{system_prompt}\n\n{user_prompt}"

    def self_correct(self, query: str, answer: str, contexts: List[Dict]) -> Dict:
        context_text = " ".join([c["text"] for c in contexts]).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())
        overlap = len(answer_words & context_words)
        confidence = min(1.0, overlap / max(len(answer_words), 1))
        return {
            "confidence": round(confidence, 2),
            "supported": confidence > 0.4,
            "needs_rephrase": confidence < 0.3,
        }

    def run(self, query: str) -> Dict:
        plan = self.plan(query)
        all_contexts = []
        for sub_q in plan["sub_queries"]:
            ctxs = self.retrieve_tool(sub_q)
            all_contexts.extend(ctxs)

        seen = set()
        unique_ctxs = []
        for c in all_contexts:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique_ctxs.append(c)
        unique_ctxs = unique_ctxs[: self.config["top_k"]]

        generation = self.generate(query, unique_ctxs)
        correction = self.self_correct(query, generation if isinstance(generation, str) else generation, unique_ctxs)

        return {
            "query": query,
            "plan": plan,
            "retrieved_contexts": unique_ctxs,
            "prompt": generation,
            "confidence": correction["confidence"],
            "supported": correction["supported"],
            "needs_rephrase": correction["needs_rephrase"],
        }


# ── Main Pipeline ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ROCKET RAG Pipeline - ChromaDB + Custom Python")
    print("=" * 60)

    config = build_config()

    Path(config["index_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["chunks_dir"]).mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Ingesting documents from Reports/ ...")
    chunks = ingest_documents(config)
    if not chunks:
        print("X No documents found. Add supported files to Reports/ and re-run.")
        return

    print("\n[2/5] Embedding & storing in ChromaDB ...")
    collection = embed_and_store(chunks, config)

    print("\n[3/5] Initializing agentic RAG agent ...")
    agent = SimpleRAGAgent(config, collection)

    print("\n[4/5] Agent ready. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("SEARCH Your question: ").strip()
            if query.lower() in ("exit", "quit", "q"):
                break
            if not query:
                continue

            result = agent.run(query)

            print(f"\nCHART Plan: {result['plan']['strategy']}")
            print(f"DOWNLOAD Retrieved {len(result['retrieved_contexts'])} chunks")
            print(f"TARGET Confidence: {result['confidence']}")
            print(f"OK Supported: {result['supported']}")
            print(f"EMBED Needs rephrase: {result['needs_rephrase']}")
            print(f"\nNOTE Generated Prompt:\n{result['prompt']}\n")
            print(f"\n{'=' * 60}\n")

        except KeyboardInterrupt:
            break

    print("\nOK Pipeline complete. Goodbye!")


if __name__ == "__main__":
    main()