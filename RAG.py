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
        self._build_token_map()

    def _build_token_map(self):
        """Pre-calculate token map for faster O(1) keyword search."""
        self.token_map = [] # List of dicts: map token -> count for each chunk
        for text in self.chunks:
            # Simple tokenization: lower and split on non-alphanumeric
            tokens = re.findall(r"\w+", text.lower())
            counts = {}
            for t in tokens:
                if len(t) >= 3:
                    counts[t] = counts.get(t, 0) + 1
            self.token_map.append(counts)
        print(f"✓ Token map built for {len(self.chunks)} chunks")
    
    @classmethod
    def from_index(cls, index_path: str, metadata_path: str, chunks_path: str,
                   embedding_model: str = 'all-MiniLM-L6-v2', top_k: int = 5) -> "RobustRAGSystem":
        """
        Create a RAG instance from a prebuilt FAISS index and saved metadata/chunks.
        """
        # Create instance without calling __init__ directly to avoid re-chunking
        instance = cls.__new__(cls)
        instance.api_key = "DEEPSEEK_ONLY"
        instance.embedding_model = SentenceTransformer(embedding_model)
        instance.top_k = top_k
        instance.faiss_dim = 384
        instance.history = []
        instance.default_law = None
        
        # Load index, metadata, and chunks
        instance.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            instance.metadata = json.load(f)
        with open(chunks_path, "r", encoding="utf-8") as f:
            instance.chunks = json.load(f)
            
        instance.embeddings = None
        instance._build_token_map() # ESSENTIAL for optimized search
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
        Uses a hybrid approach: keyword matching + semantic search.
        """
        q = (query or "").lower()
        
        # Build keyword set from query
        tokens = [t for t in re.findall(r"\w+", q) if len(t) >= 3]
        token_set = set(tokens)
        
        keyword_scores: List[Tuple[float, int]] = []
        if token_set:
            if hasattr(self, "token_map") and self.token_map:
                # OPTIMIZED: Use pre-calculated token map
                for i, counts in enumerate(self.token_map):
                    score = 0.0
                    for tok in token_set:
                        count = counts.get(tok, 0)
                        if count > 0:
                            score += count * (1 + len(tok) / 10)
                    if score > 0:
                        keyword_scores.append((score, i))
            else:
                # Fallback for old instances
                for i, text in enumerate(self.chunks):
                    t = text.lower()
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
        """Generate response and return both text and sources for internal use."""
        chunk_texts, chunk_metas = self.retrieve_for_generation(query)
        if not chunk_texts:
            return self._get_empty_result(), []
        
        # Prepare context and prompt
        messages, references = self._prepare_llm_input(query, chunk_texts, chunk_metas)
        
        # Sync completion
        ds_prompt = "System:\n" + messages[0]["content"] + "\n\n" + "\n\n".join([m["content"] for m in messages[1:]])
        analysis = openai_completion(ds_prompt, model="deepseek-chat", temperature=0.3, max_tokens=800)
        
        # Combine with sources
        answer = analysis.strip() + self._format_sources_section(references)
        
        # Update history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        return answer, chunk_metas

    def generate_response_stream(self, query):
        """Generator that yields chunks of the response in real-time."""
        chunk_texts, chunk_metas = self.retrieve_for_generation(query)
        if not chunk_texts:
            yield self._get_empty_result()
            return

        messages, references = self._prepare_llm_input(query, chunk_texts, chunk_metas)
        ds_prompt = "System:\n" + messages[0]["content"] + "\n\n" + "\n\n".join([m["content"] for m in messages[1:]])
        
        # Streaming completion
        client = initialize_openai_client()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": ds_prompt}],
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=800,
            stream=True
        )
        
        full_text = ""
        for part in response:
            delta = part.choices[0].delta.content or ""
            full_text += delta
            yield delta
            
        # Append sources at the end
        sources_text = self._format_sources_section(references)
        yield sources_text
        
        # Update history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": full_text + sources_text})

    def _get_empty_result(self) -> str:
        return """Je n'ai pas trouvé d'informations pertinentes pour cette question dans la base de données juridique.

**Suggestions:**
- Reformulez votre question avec des termes juridiques précis
- Mentionnez le domaine de droit concerné (pénal, civil, fiscal, etc.)
- Essayez une recherche plus spécifique (ex: "sanctions pour vol" au lieu de "vol")

*Cet outil est développé par Pierre Guy A. NJOCK - Des améliorations continues sont en cours.*"""

    def _prepare_llm_input(self, query, chunk_texts, chunk_metas):
        # Refactored common logic for prompt building
        references = []
        ref_map = {}
        context_parts = []
        for i, meta in enumerate(chunk_metas):
            art = meta.get("article", "Article inconnu")
            law = meta.get("law", "Loi inconnue")
            ref_key = f"{art} - {law}"
            if ref_key not in ref_map:
                ref_num = len(references) + 1
                ref_map[ref_key] = ref_num
                references.append({"num": ref_num, "article": art, "law": law, "is_primary": i == 0})
            ref_num = ref_map[ref_key]
            context_parts.append(f"[{ref_num}] {art} ({law}):\n{chunk_texts[i]}")
        
        joined_context = "\n\n---\n\n".join(context_parts)
        
        # Re-use the system_prompt snippet but shortened logic for brevity in code
        system_prompt = self._get_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        
        # Include history
        recent = self.history[-6:]
        for msg in recent:
            messages.append(msg) # Simplification: assume history is filtered if needed

        user_content = f"Question: {query}\n\nRetrieved Legal Context:\n\n{joined_context}\n\nProvide a structured legal analysis based on the above context."
        messages.append({"role": "user", "content": user_content})
        return messages, references

    def _get_system_prompt(self):
        return """You are an expert legal-analytical assistant specialized in Cameroonian law.

You answer questions EXCLUSIVELY using the provided retrieved context [extracted articles].
General legal knowledge may be used ONLY to explain or clarify the retrieved context,
never to introduce new legal rules, obligations, rights, or interpretations.

If the answer is not fully supported by the retrieved context, you MUST say so explicitly.

For every user question, follow this process internally before producing the final answer:

────────────────────────────────
1. Context Verification & Scoping
────────────────────────────────
- Identify which parts of the retrieved context are relevant to the question.
- Confirm that the context pertains to Cameroonian law.
- Determine whether the retrieved context is sufficient to answer the question.
- If the context is insufficient, contradictory, or silent:
  • State this clearly in the final answer.
  • Do NOT infer, extrapolate, or rely on external legal knowledge.

────────────────────────────
2. Question Augmentation
────────────────────────────
- Restate the question in richer form by identifying:
  • The explicit legal issue being asked
  • Implicit assumptions (e.g., applicable code, time period, legal status)
  • Missing but relevant legal context (if absent from retrieval)
  • Likely user intent (informational, compliance, interpretation)
- Resolve ambiguities only when the retrieved context allows it.
- Do NOT ask clarification questions unless the ambiguity prevents a legally correct answer.

────────────────────────────
3. Decomposition
────────────────────────────
- Break the augmented question into legally meaningful sub-issues.
- Classify each sub-issue as:
  • Doctrinal (what the law states)
  • Interpretive (how provisions relate)
  • Procedural (how the law is applied)
  • Practical effect (legal consequences)
- Order sub-issues logically, following the structure of the law where possible.

────────────────────────────
4. Context-Grounded Analysis
────────────────────────────
For each sub-issue:
- Base the explanation strictly on the retrieved legal text.
- Quote or paraphrase the law accurately.
- Identify relevant articles, sections, or provisions when available.
- State assumptions explicitly (e.g., applicability conditions).
- Note limits of interpretation where the text is silent or ambiguous.
- Avoid policy opinions, speculation, or comparative law unless explicitly present in context.

────────────────────────────
5. Synthesis
────────────────────────────
- Integrate the sub-answers into a coherent legal explanation.
- Ensure internal consistency with the retrieved legal materials.
- Highlight how different provisions interact, if supported by context.
- Distinguish clearly between:
  • What the law explicitly states
  • What follows directly from the text
  • What cannot be determined from the context

────────────────────────────
6. Structured Output
────────────────────────────
- Present the final answer using:
  • Clear legal section headers
  • Bullet points or numbered steps where appropriate
  • Progressive depth (overview → provisions → implications)
- Reference relevant legal articles using [1], [2], etc. matching the provided context numbers.
- Use precise legal terminology consistent with the source text.

────────────────────────────
7. Style Constraints
────────────────────────────
- Be concise, precise, and legally neutral.
- Avoid metaphors, rhetoric, or persuasive language.
- Use plain, formal legal language suitable for non-specialists.
- Do not provide legal advice beyond what is explicitly stated in the law.
- Respond in the same language as the user's question (French or English).

────────────────────────────
8. Quality Control & Safety
────────────────────────────
- Verify that every legal claim is supported by the retrieved context.
- Ensure no external legal rules or assumptions are introduced.
- If the context does not answer the question:
  • State: "The provided legal texts do not address this issue."
- Never hallucinate articles, rights, obligations, or procedures.

Output ONLY the final structured answer.
Do not reveal internal reasoning, chain-of-thought, or intermediate analysis."""


    def _format_sources_section(self, references):
        primary_refs = [r for r in references if r["is_primary"]]
        supplementary_refs = [r for r in references if not r["is_primary"]]
        sources_section = "\n\n---\n\n**📚 Sources:**\n\n"
        if primary_refs:
            p = primary_refs[0]
            sources_section += f"**Source principale:**\n[{p['num']}] {p['article']} - *{p['law']}*\n\n"
        if supplementary_refs:
            sources_section += "**Sources complémentaires:**\n"
            for r in supplementary_refs[:5]:
                sources_section += f"[{r['num']}] {r['article']} - *{r['law']}*\n"
            sources_section += "\n"
        sources_section += "**Articles contradictoires:** Aucun identifié dans le contexte fourni."
        return sources_section



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

