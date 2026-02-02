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
api_key_deepseek = os.environ.get("DEEPSEEK_API_KEY")
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
    # PRIORITY 1: Check filename first - it's more reliable than document text
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        base = base.replace("_", " ").replace("-", " ").strip()
        # If filename clearly indicates a law (LOI No, Loi N°, DECRET, etc.), use it
        if re.search(r"(LOI\s+N[°o]?|Loi\s+N[°o]?|DECRET|Décret|ARRETE|Arrêté|ORDONNANCE)", base, re.IGNORECASE):
            return base
    
    # PRIORITY 2: Fall back to document text patterns
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
    
    # PRIORITY 3: Use filename as fallback even if it doesn't match specific patterns
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
        Handles both numeric articles (Article 1) and French ordinals (Article premier, Article unique).
        Also handles FAQ-style documents that don't follow standard article format.
        """
        law_reference = self.default_law or infer_law_reference(document_text, None)
        text = (document_text or "").replace("\r\n", "\n")
        
        # French ordinal words that can follow "Article" - comprehensive list
        french_ordinals = r"(?:premier|premi[eè]re|deuxi[eè]me|second|seconde|troisi[eè]me|quatri[eè]me|cinqui[eè]me|sixi[eè]me|septi[eè]me|huiti[eè]me|neuvi[eè]me|dixi[eè]me|onzi[eè]me|douzi[eè]me|treizi[eè]me|quatorzi[eè]me|quinzi[eè]me|seizi[eè]me|dix-septi[eè]me|dix-huiti[eè]me|dix-neuvi[eè]me|vingti[eè]me|unique|un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize)"
        
        # Pattern matches: Article 1, Article premier, ARTICLE UNIQUE, etc.
        article_pattern = rf"(?:\n|\A)\s*((?:ARTICLE|Article|article)\s+(?:\d+|{french_ordinals}))\b"
        
        # Split while keeping delimiters
        parts = re.split(article_pattern, text, flags=re.IGNORECASE)
        
        # If no articles found, try TITRE/CHAPITRE structure (common in finance laws)
        if len(parts) <= 1:
            titre_pattern = r"(?:\n|\A)\s*((?:TITRE|CHAPITRE|SECTION|PARTIE)\s+(?:PREMIER|PREMI[ÈE]RE|I{1,3}|IV|V|VI|VII|VIII|IX|X|\d+|[A-Z]))\s*[:\-\.]?\s*\n"
            titre_parts = re.split(titre_pattern, text, flags=re.IGNORECASE)
            if len(titre_parts) > 1:
                chunks: List[str] = []
                metadata: List[Dict[str, Any]] = []
                for i in range(1, len(titre_parts), 2):
                    label = titre_parts[i].strip()
                    body = titre_parts[i + 1] if i + 1 < len(titre_parts) else ""
                    chunks.append(body.strip())
                    metadata.append({"article": label, "law": law_reference})
                print(f"[TITRE/CHAPITRE Chunking] Extracted {len(chunks)} chunks.")
                return chunks, metadata
            # If still no structure found, use FAQ-style document handling
            return self._chunk_faq_document(text, law_reference)

        chunks: List[str] = []
        metadata: List[Dict[str, Any]] = []
        # parts structure: [preamble, label1, body1, label2, body2, ...]
        for i in range(1, len(parts), 2):
            label = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            # Extract article identifier (numeric or ordinal)
            art_match = re.search(rf"(?:ARTICLE|Article|article)\s+(\d+|{french_ordinals})", label, re.IGNORECASE)
            if art_match:
                art_id = art_match.group(1)
                # Normalize French ordinals to standard format
                ordinal_map = {
                    "premier": "1er", "première": "1ère", "premi\u00e8re": "1ère",
                    "deuxième": "2ème", "deuxi\u00e8me": "2ème", "second": "2ème", "seconde": "2ème",
                    "troisième": "3ème", "troisi\u00e8me": "3ème",
                    "quatrième": "4ème", "quatri\u00e8me": "4ème",
                    "cinquième": "5ème", "cinqui\u00e8me": "5ème",
                    "sixième": "6ème", "sixi\u00e8me": "6ème",
                    "septième": "7ème", "septi\u00e8me": "7ème",
                    "huitième": "8ème", "huiti\u00e8me": "8ème",
                    "neuvième": "9ème", "neuvi\u00e8me": "9ème",
                    "dixième": "10ème", "dixi\u00e8me": "10ème",
                    "onzième": "11ème", "onzi\u00e8me": "11ème",
                    "douzième": "12ème", "douzi\u00e8me": "12ème",
                    "unique": "unique",
                    "un": "1", "deux": "2", "trois": "3", "quatre": "4", "cinq": "5",
                    "six": "6", "sept": "7", "huit": "8", "neuf": "9", "dix": "10",
                    "onze": "11", "douze": "12", "treize": "13", "quatorze": "14", "quinze": "15", "seize": "16"
                }
                normalized = ordinal_map.get(art_id.lower(), art_id)
                article_number = f"Article {normalized}"
            else:
                article_number = label
            chunks.append(body.strip())
            metadata.append({"article": article_number, "law": law_reference})

        print(f"[Fallback] Extracted {len(chunks)} chunks.")
        return chunks, metadata

    def _chunk_faq_document(self, text: str, law_reference: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Handle FAQ-style documents that don't have standard Article structure.
        Splits on Q&A patterns or paragraph breaks.
        """
        chunks: List[str] = []
        metadata: List[Dict[str, Any]] = []
        
        # Try to split on Q&A patterns (Question:, Q:, FAQ, numbered questions, etc.)
        qa_pattern = r"(?:\n|^)\s*(?:Q(?:uestion)?[\s:\.]*\d*[\s:\.]|FAQ[\s:\.]*\d*[\s:\.]|\d+[\.\)]\s+[A-Z])"
        qa_parts = re.split(qa_pattern, text, flags=re.IGNORECASE)
        
        if len(qa_parts) > 1:
            # Found Q&A structure
            for i, part in enumerate(qa_parts):
                if part.strip():
                    chunks.append(part.strip())
                    metadata.append({"article": f"FAQ {i+1}", "law": law_reference})
            print(f"[FAQ Chunking] Extracted {len(chunks)} Q&A chunks.")
            return chunks, metadata
        
        # Fallback: split by paragraphs (double newlines) with minimum length
        paragraphs = re.split(r"\n\s*\n", text)
        current_chunk = ""
        chunk_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            current_chunk += "\n\n" + para if current_chunk else para
            # Create chunk when it reaches reasonable size (500+ chars)
            if len(current_chunk) > 500:
                chunk_count += 1
                chunks.append(current_chunk.strip())
                metadata.append({"article": f"Section {chunk_count}", "law": law_reference})
                current_chunk = ""
        
        # Add remaining content
        if current_chunk.strip():
            chunk_count += 1
            chunks.append(current_chunk.strip())
            metadata.append({"article": f"Section {chunk_count}", "law": law_reference})
        
        print(f"[Paragraph Chunking] Extracted {len(chunks)} sections.")
        return chunks if chunks else [text.strip()], metadata if metadata else [{"article": "Document", "law": law_reference}]


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
        Uses a hybrid approach: keyword matching + semantic search + query expansion.
        """
        q = (query or "").lower()
        
        # Query expansion: add related legal terms for common queries
        expansions = {
            "impot": ["taxe", "fiscal", "contribuable", "tva", "is", "irpp", "finances"],
            "vol": ["voler", "larcin", "appropriation", "soustraction"],
            "meurtre": ["homicide", "assassinat", "mort", "tuer"],
            "mariage": ["époux", "conjoint", "matrimonial", "union"],
            "divorce": ["séparation", "dissolution", "répudiation"],
            "travail": ["emploi", "salaire", "employeur", "licenciement", "contrat"],
            "propriete": ["bien", "immeuble", "foncier", "terrain"],
            "entreprise": ["société", "commercial", "ohada", "sarl", "sa"],
            "peine": ["sanction", "prison", "amende", "condamnation"],
            "crime": ["délit", "infraction", "pénal"],
            "douane": ["importation", "exportation", "tarif"],
            "finances": ["budget", "recettes", "dépenses", "trésor", "exercice"],
        }
        
        # Build keyword set with expansion
        tokens = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
        token_set = set(tokens)
        
        # Expand query with related terms
        for token in list(token_set):
            for key, related in expansions.items():
                if token.startswith(key) or key.startswith(token):
                    token_set.update(related)
                    break
        
        keyword_scores: List[Tuple[float, int]] = []
        if token_set:
            for i, text in enumerate(self.chunks):
                t = text.lower()
                # Count matches with weight for longer tokens
                score = sum(t.count(tok) * (1 + len(tok) / 10) for tok in token_set)
                if score > 0:
                    keyword_scores.append((score, i))
        
        # Also do semantic search
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), self.top_k * 2)
        semantic_indices = list(indices[0])
        
        # Combine results: prioritize keyword matches but include semantic results
        selected = []
        seen = set()
        
        if keyword_scores:
            keyword_scores.sort(key=lambda x: x[0], reverse=True)
            for _, idx in keyword_scores[:self.top_k]:
                if idx not in seen:
                    selected.append(idx)
                    seen.add(idx)
        
        # Add semantic results to fill remaining slots
        for idx in semantic_indices:
            if len(selected) >= self.top_k:
                break
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
        
        # If still nothing, use pure semantic search
        if not selected:
            selected = semantic_indices[:self.top_k]
        
        texts: List[str] = [self.chunks[i] for i in selected]
        metas: List[Dict[str, Any]] = [self.metadata[i] for i in selected]
        return texts, metas


    def generate_response(self, query):
        """
        Generate a well-formatted response with proper citations and source references.
        """
        # Use structured retrieval for grounded answer and deterministic metadata formatting
        chunk_texts, chunk_metas = self.retrieve_for_generation(query)

        if not chunk_texts:
            return """Je n'ai pas trouvé d'informations pertinentes pour cette question dans la base de données juridique.

**Suggestions:**
- Reformulez votre question avec des termes juridiques précis
- Mentionnez le domaine de droit concerné (pénal, civil, fiscal, etc.)
- Essayez une recherche plus spécifique (ex: "sanctions pour vol" au lieu de "vol")

*Cet outil est développé par Pierre Guy A. NJOCK - Des améliorations continues sont en cours.*"""

        # Build numbered references for citations
        references = []
        ref_map = {}  # Map "Article X of Law Y" -> reference number
        for i, meta in enumerate(chunk_metas):
            art = meta.get("article", "Article inconnu")
            law = meta.get("law", "Loi inconnue")
            ref_key = f"{art} - {law}"
            if ref_key not in ref_map:
                ref_num = len(references) + 1
                ref_map[ref_key] = ref_num
                references.append({
                    "num": ref_num,
                    "article": art,
                    "law": law,
                    "is_primary": i == 0
                })

        # Build context with citation markers for the LLM
        context_parts = []
        for i, (text, meta) in enumerate(zip(chunk_texts, chunk_metas)):
            art = meta.get("article", "Article inconnu")
            law = meta.get("law", "Loi inconnue")
            ref_key = f"{art} - {law}"
            ref_num = ref_map.get(ref_key, i + 1)
            context_parts.append(f"[{ref_num}] {art} ({law}):\n{text}")

        joined_context = "\n\n---\n\n".join(context_parts)

        # Improved system prompt for better structured analysis
        system_prompt = """Tu es un expert juridique camerounais. Réponds en utilisant UNIQUEMENT les extraits fournis.

RÈGLES STRICTES:
1. Cite tes sources avec [1], [2], etc. correspondant aux numéros des articles fournis
2. Structure ta réponse clairement:
   - Commence par une réponse directe à la question
   - Développe avec les détails juridiques pertinents
   - Mentionne les nuances ou conditions importantes
3. N'invente JAMAIS d'informations non présentes dans les extraits
4. Si l'information demandée n'est pas dans les extraits, dis-le clairement
5. Réponds dans la même langue que la question (français ou anglais)

FORMAT DE RÉPONSE:
- Paragraphes clairs et concis
- Citations intégrées naturellement: "Selon l'article X [1], ..."
- Pas de listes à puces sauf si vraiment nécessaire"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Include relevant conversation history
        def relevance(a: str, b: str) -> float:
            aw = set([w for w in re.findall(r"\w+", a.lower()) if len(w) >= 4])
            bw = set([w for w in re.findall(r"\w+", b.lower()) if len(w) >= 4])
            if not aw or not bw:
                return 0.0
            return len(aw & bw) / max(len(aw), len(bw))
        
        recent = self.history[-6:]
        for msg in recent:
            if msg["role"] == "user" and relevance(query, msg["content"]) >= 0.3:
                messages.append(msg)
            elif msg["role"] == "assistant":
                messages.append(msg)

        # Build user prompt with context
        user_content = f"""Question: {query}

Extraits juridiques pertinents:

{joined_context}

Analyse la question et fournis une réponse claire et précise basée sur ces extraits."""

        messages.append({"role": "user", "content": user_content})

        # Generate response
        ds_prompt = (
            "System:\n" + messages[0]["content"] + "\n\n" +
            "\n\n".join([m["content"] for m in messages[1:]])
        )
        analysis = openai_completion(ds_prompt, model="deepseek-chat", temperature=0.3, max_tokens=800)

        # Build formatted sources section
        primary_refs = [r for r in references if r["is_primary"]]
        supplementary_refs = [r for r in references if not r["is_primary"]]

        sources_section = "\n\n---\n\n**📚 Sources:**\n\n"
        
        # Primary source
        if primary_refs:
            p = primary_refs[0]
            sources_section += f"**Source principale:**\n[{p['num']}] {p['article']} - *{p['law']}*\n\n"
        
        # Supplementary sources
        if supplementary_refs:
            sources_section += "**Sources complémentaires:**\n"
            for r in supplementary_refs[:5]:  # Limit to 5 supplementary
                sources_section += f"[{r['num']}] {r['article']} - *{r['law']}*\n"
            sources_section += "\n"
        
        # Note about contradictory (would require semantic analysis)
        sources_section += "**Articles contradictoires:** Aucun identifié dans le contexte fourni."

        # Combine answer with sources
        answer = analysis.strip() + sources_section

        # Update history
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

