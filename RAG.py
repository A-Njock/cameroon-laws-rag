import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import logging
import os
import re
from typing import List, Tuple, Dict, Any
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepSeek configuration
api_key_deepseek = "sk-941d68a6421e4c3cb2bb17f4e53d258a"
base_url_deepseek = "https://api.deepseek.com/v1"


def initialize_openai_client() -> OpenAI:
    client = OpenAI(api_key=api_key_deepseek, base_url=base_url_deepseek)
    return client


def openai_completion(prompt: str, model: str = "deepseek-chat", temperature: float = 0.2, max_tokens: int = 4096):
    client = initialize_openai_client()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def normalize_text_for_patterns(text: str) -> str:
    return (text or "").replace("\u00A0", " ").replace("Nº", "N°").replace("No", "N°")


def infer_law_reference(document_text: str, filename: str | None) -> str:
    text = normalize_text_for_patterns(document_text)
    head = "\n".join(text.splitlines()[:80])
    patterns = [
        r"(Loi\s+N[°o]\s*[\w/\-]+.*)",
        r"(Loi\s+.*?Code\s+P[ée]nal.*)",
        r"(D[ée]cret\s+N[°o]?\s*[\w/\-]+.*)",
        r"(D[ée]cret\s+.*)",
        r"(Arr[ée]t[ée]\s+N[°o]?\s*[\w/\-]+.*)",
        r"(Arr[ée]t[ée]\s+.*)",
        r"(Ordonnance\s+N[°o]?\s*[\w/\-]+.*)",
        r"(Ordonnance\s+.*)",
        r"(Code\s+P[ée]nal.*)",
    ]
    for pat in patterns:
        m = re.search(pat, head, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        base = base.replace("_", " ").replace("-", " ").strip()
        if re.search(r"(loi|décret|decret|arr[ée]t[ée]|ordonnance|code)", base, re.IGNORECASE):
            return base
    return "Unknown law reference"


class RobustRAGSystem:
    def __init__(self, api_key, document_text, embedding_model='all-MiniLM-L6-v2', top_k=5, faiss_dim=384, default_law: str | None = None):
        self.api_key = api_key
        self.embedding_model = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.faiss_dim = faiss_dim
        self.history = []
        self.default_law = default_law

        self.chunks, self.metadata = self._intelligent_chunking(document_text)
        self._build_vector_store()
    
    @classmethod
    def from_index(cls, index_path: str, metadata_path: str, chunks_path: str,
                   embedding_model: str = 'all-MiniLM-L6-v2', top_k: int = 5) -> "RobustRAGSystem":
        """
        Create a RAG instance from a prebuilt FAISS index and saved metadata/chunks.
        """
        # Create a dummy instance without building from a document
        instance = cls.__new__(cls)
        instance.api_key = "DEEPSEEK_ONLY"
        instance.embedding_model = SentenceTransformer(embedding_model)
        instance.top_k = top_k
        instance.faiss_dim = None
        instance.history = []
        instance.default_law = None
        # Load index, metadata, and chunks
        instance.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            instance.metadata = json.load(f)
        with open(chunks_path, "r", encoding="utf-8") as f:
            instance.chunks = json.load(f)
        # Embeddings not required; FAISS index already built
        instance.embeddings = None
        return instance

    def _intelligent_chunking(self, document_text):
        """
        Uses an LLM to intelligently chunk the document into articles, keeping track of article numbers and law sections.
        """
        prompt = """
You are an expert legal document parser. Your task is to extract individual articles from the provided Cameroon law document (Penal Code). 

For each article, identify:
- The article number (e.g., "Article 1").
- The full text of the article.
- The law reference (e.g., "Loi N°2016/007 du 12 juillet 2016 PORTANT CODE PENAL").

Return the output as a **valid JSON list of objects**, with no extra formatting, newlines, or quotes outside the JSON structure. Each object must be:
{{
  "article": "Article number",
  "law": "Law reference",
  "text": "Full article text"
}}

If the document is incomplete or truncated, extract what is available. Do not add or invent any content.

Document text:
{document_text}
""".format(document_text=document_text)

        try:
            raw_response = openai_completion(prompt, model="deepseek-chat", temperature=0.2, max_tokens=2000)
            if not raw_response or not raw_response.strip():
                raise ValueError("Empty response from DeepSeek")
            parsed_json = json.loads(raw_response)
            if not isinstance(parsed_json, list):
                raise ValueError("Expected a JSON list.")
        except Exception as e:
            print(f"Structured chunking failed ({e}). Raw response: {str(raw_response)[:300] if 'raw_response' in locals() else 'None'}")
            # Fallback to heuristic chunking
            return self._fallback_chunking(document_text)

        chunks = [item.get('text', '').strip() for item in parsed_json]
        metadata: List[Dict[str, Any]] = []
        for item in parsed_json:
            meta = {k: v for k, v in item.items() if k != 'text'}
            if not meta.get('law'):
                meta['law'] = self.default_law or infer_law_reference(document_text, None)
            metadata.append(meta)

        print(f"Extracted {len(chunks)} chunks.")
        return chunks, metadata

    def _fallback_chunking(self, document_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Heuristic fallback: split on ARTICLE headings; attempt to infer article number and use default_law.
        """
        law_reference = self.default_law or infer_law_reference(document_text, None)
        text = (document_text or "").replace("\r\n", "\n")
        # Split while keeping delimiters
        parts = re.split(r"(?:\n|\A)\s*((?:ARTICLE|Article)\s+\d+)\b", text)
        if len(parts) <= 1:
            return [text.strip()], [{"article": "Unknown", "law": law_reference}]

        chunks: List[str] = []
        metadata: List[Dict[str, Any]] = []
        # parts structure: [preamble, label1, body1, label2, body2, ...]
        for i in range(1, len(parts), 2):
            label = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            art_num_match = re.search(r"(?:ARTICLE|Article)\s+(\d+)", label)
            article_number = f"Article {art_num_match.group(1)}" if art_num_match else label
            chunks.append(body.strip())
            metadata.append({"article": article_number, "law": law_reference})

        print(f"[Fallback] Extracted {len(chunks)} chunks.")
        return chunks, metadata

    def _build_vector_store(self):
        embeddings = self.embedding_model.encode(self.chunks)
        self.index = faiss.IndexFlatL2(self.faiss_dim)
        self.index.add(np.array(embeddings).astype('float32'))
        self.embeddings = embeddings

    def retrieve_chunks(self, query):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), self.top_k)

        retrieved = []
        for idx in indices[0]:
            chunk = self.chunks[idx]
            meta = self.metadata[idx]
            law = meta.get('law', 'Unknown law')
            article = meta.get('article', 'Unknown article')
            chapter = meta.get('chapter')  # optional
            if chapter:
                enhanced_chunk = f"Law: {law}\nChapter: {chapter}\nArticle: {article}\n{chunk}"
            else:
                enhanced_chunk = f"Law: {law}\nArticle: {article}\n{chunk}"
            retrieved.append(enhanced_chunk)

        return retrieved

    def retrieve_for_generation(self, query) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Return plain chunk texts and their metadata for grounded generation.
        Strategy:
        1) Try exact/keyword match ranking across all chunks.
        2) If nothing matches, fall back to embedding search.
        """
        q = (query or "").lower()
        # Build keyword set: words of length >= 4
        tokens = [t for t in re.findall(r"\w+", q) if len(t) >= 4]
        token_set = set(tokens)
        keyword_scores: List[Tuple[int, int]] = []  # (score, idx)
        if token_set:
            for i, text in enumerate(self.chunks):
                t = text.lower()
                score = sum(t.count(tok) for tok in token_set)
                if score > 0:
                    keyword_scores.append((score, i))
        if keyword_scores:
            # Sort by score desc, take top_k
            keyword_scores.sort(key=lambda x: x[0], reverse=True)
            selected = [idx for _, idx in keyword_scores[: self.top_k]]
        else:
            # Fallback to embedding search
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.index.search(np.array(query_embedding).astype('float32'), self.top_k)
            selected = list(indices[0])
        texts: List[str] = [self.chunks[i] for i in selected]
        metas: List[Dict[str, Any]] = [self.metadata[i] for i in selected]
        return texts, metas

    def generate_response(self, query):
        # Use structured retrieval for grounded answer and deterministic metadata formatting
        chunk_texts, chunk_metas = self.retrieve_for_generation(query)

        if not chunk_texts:
            return "Cet Outils a ete cree par Pierre Guy A.Njock, Des travaux sont cours pour ameliorer la reponse a cette question particuliere. En attendant, Cliquer sur le lien pour voir nos formation sur comment gagner de l'argent grace a l'IA ( /formations )"

        # Instruct the model to write only a single coherent paragraph. No lists, no disclaimers.
        system_prompt = """
You are a legal expert. Answer ONLY using the provided document chunks.
Write ONE concise, coherent paragraph directly answering the user's question.
Do NOT invent any information not present in the chunks. If a specific detail is not in the chunks, omit it.
Do NOT include headings, lists, citations, or disclaimers. Paragraph only.
Respond in the same language as the user input.
"""

        messages = [{"role": "system", "content": system_prompt}]
        # Include only relevant past turns based on simple token overlap
        def relevance(a: str, b: str) -> float:
            aw = set([w for w in re.findall(r"\w+", a.lower()) if len(w) >= 4])
            bw = set([w for w in re.findall(r"\w+", b.lower()) if len(w) >= 4])
            if not aw or not bw:
                return 0.0
            inter = len(aw & bw)
            denom = max(len(aw), len(bw))
            return inter / denom
        recent = self.history[-10:]
        for msg in recent:
            if msg["role"] == "user":
                if relevance(query, msg["content"]) >= 0.3:
                    messages.append(msg)
            elif msg["role"] == "assistant":
                # Tie assistant message relevance to the last user message, if available
                messages.append(msg)

        # Provide only the raw chunk texts to tighten grounding
        joined_chunks = "\n\n---\n\n".join(chunk_texts)
        user_content = "Query: " + query + "\n\nRelevant Chunks:\n" + joined_chunks
        messages.append({"role": "user", "content": user_content})

        ds_prompt = (
            "System:\n" + messages[0]["content"] + "\n\n" +
            "\n\n".join([m["content"] for m in messages[1:]])
        )
        paragraph = openai_completion(ds_prompt, model="deepseek-chat", temperature=0.3, max_tokens=600)

        # Build exact article and similar articles deterministically from retrieved metadata
        primary_meta = chunk_metas[0] if chunk_metas else {}
        exact_article = primary_meta.get("article", "Unknown")
        exact_law = primary_meta.get("law", "Unknown")
        exact_line = f"**Exact Article Number:** {exact_article} of {exact_law}."

        # Complementary: list distinct other articles from remaining metas
        other_articles = []
        for m in chunk_metas[1:]:
            art = m.get("article")
            law = m.get("law")
            if art and law:
                formatted = f"{art} of {law}"
            elif art:
                formatted = art
            elif law:
                formatted = law
            else:
                continue
            if formatted not in other_articles:
                other_articles.append(formatted)
        if other_articles:
            complementary_line = "-   **Complementary:** " + "; ".join(other_articles) + "."
        else:
            complementary_line = "-   **Complementary:** None found in the provided context."
        contradictory_line = "-   **Contradictory:** None found in the provided context."

        answer = paragraph.strip() + "\n\n" + exact_line + "\n\n**Similar Articles:**\n" + complementary_line + "\n\n" + contradictory_line

        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})

        return answer


def extract_text_from_pdf(pdf_path: str) -> str:
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required to read PDFs. Please install it: pip install PyPDF2")
    text_parts: List[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_idx, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
            except Exception:
                extracted = ""
            text_parts.append(extracted)
    return "\n".join(text_parts)


def build_faiss_from_folder(folder_path: str, index_output_path: str, metadata_output_path: str,
                            embedding_model: str = 'all-MiniLM-L6-v2') -> None:
    print(f"Building FAISS index from folder: {folder_path}")
    model = SentenceTransformer(embedding_model)
    all_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(root, filename)
            print(f"\nProcessing file: {pdf_path}")
            try:
                doc_text = extract_text_from_pdf(pdf_path)
            except Exception as e:
                print(f"Failed to read PDF '{pdf_path}': {e}")
                continue

            doc_level_law = infer_law_reference(doc_text, pdf_path)
            print(f"Inferred law reference: {doc_level_law}")

            rag = RobustRAGSystem(
                api_key="DEEPSEEK_ONLY",
                document_text=doc_text,
                embedding_model=embedding_model,
                default_law=doc_level_law
            )
            for i, (chunk_text, meta) in enumerate(zip(rag.chunks, rag.metadata), start=1):
                preview = chunk_text[:200].replace("\n", " ")
                print(f"- Chunk {i}: {meta.get('article', 'Unknown')} | Law: {meta.get('law', 'Unknown')} | Preview: {preview}...")

            all_chunks.extend(rag.chunks)
            all_metadata.extend(rag.metadata)

    if not all_chunks:
        print("No chunks extracted. Aborting index build.")
        return

    print(f"\nEncoding {len(all_chunks)} total chunks...")
    embeddings = model.encode(all_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    print(f"Writing FAISS index to: {index_output_path}")
    faiss.write_index(index, index_output_path)

    print(f"Writing metadata to: {metadata_output_path}")
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    # Also persist chunks so RAG can answer without re-indexing
    chunks_output_path = os.path.splitext(metadata_output_path)[0] + ".chunks.json"
    print(f"Writing chunks to: {chunks_output_path}")
    with open(chunks_output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    print("Index build complete.")


if __name__ == "__main__":
    # Build FAISS index from folder "LOIS CAMEROUN" and save index + metadata
    index_path = os.path.join(os.getcwd(), "index_file.index")
    metadata_path = os.path.join(os.getcwd(), "index_file.meta.json")
    chunks_path = os.path.splitext(metadata_path)[0] + ".chunks.json"
    # build_faiss_from_folder is intentionally commented out to freeze index generation
    # folder = os.path.join(os.getcwd(), "LOIS CAMEROUN")
    # build_faiss_from_folder(folder, index_path, metadata_path)

    # Load prebuilt index and start an interactive QA loop
    if not (os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(chunks_path)):
        print("Prebuilt index or data files not found. Expected files:")
        print(f"- {index_path}")
        print(f"- {metadata_path}")
        print(f"- {chunks_path}")
        print("If missing, temporarily uncomment build_faiss_from_folder to generate them.")
    else:
        rag = RobustRAGSystem.from_index(index_path, metadata_path, chunks_path, embedding_model='all-MiniLM-L6-v2', top_k=5)
        print("RAG loaded with prebuilt index. Type a question (or just press Enter to exit).")
        try:
            while True:
                user_q = input("\nYou: ").strip()
                if not user_q:
                    break
                answer = rag.generate_response(user_q)
                print("\nAssistant:", answer)
        except KeyboardInterrupt:
            pass
        print("\nSession ended.")

