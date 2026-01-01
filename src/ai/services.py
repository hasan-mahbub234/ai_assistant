"""
AI service layer for RAG pipeline and legal document analysis
USING GROQ INSTEAD OF GEMINI
"""
import re
import json
import logging
import time
import re
from typing import Any, Dict, List, Optional

import httpx
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings

logger = logging.getLogger(__name__)


# =========================================================
# AI Processor (Singleton)
# =========================================================

class AIProcessor:
    """
    Central AI service:
    - Groq (Llama/Mixtral) → document analysis & chat
    - SentenceTransformer → embeddings
    - Pinecone → vector storage
    """

    def __init__(self):
        self._init_groq()
        self._init_embeddings()
        self._init_pinecone()
        self._init_text_splitter()

    # =====================================================
    # Groq Initialization
    # =====================================================

    def _init_groq(self):
        if not settings.GROQ_API_KEY:
            self.client = None
            logger.warning("GROQ_API_KEY not set → analysis disabled")
            return

        try:
            # Groq uses OpenAI-compatible API
            self.client = {
                "api_key": settings.GROQ_API_KEY,
                "base_url": settings.GROQ_BASE_URL,
                "model": settings.GROQ_MODEL,
                "timeout": 60.0
            }
            self.analysis_model = settings.GROQ_MODEL
            self.chat_model = settings.GROQ_MODEL
            logger.info(f"Groq client initialized with model: {settings.GROQ_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None

    # =====================================================
    # Embeddings (LOCAL ONLY – SAFE)
    # =====================================================

    def _init_embeddings(self):
        """
        Pinecone index dimension = 768
        This model outputs exactly 768 dimensions
        """
        try:
            self.embedder = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2"
            )
            self.embedding_dim = 768
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        final_vectors: List[List[float]] = []

        for vec in vectors:
            vec_list = vec.tolist()

            # Check if vector is valid
            if not any(vec_list):
                logger.warning("Generated embedding is all zeros, skipping")
                continue

            if len(vec_list) != self.embedding_dim:
                logger.error(f"Embedding dim mismatch: {len(vec_list)} != {self.embedding_dim}")
                # This shouldn't happen with the selected model
                continue

            final_vectors.append(vec_list)

        return final_vectors

    # =====================================================
    # Pinecone Initialization
    # =====================================================

    def _init_pinecone(self):
        self.index = None

        if not settings.PINECONE_API_KEY:
            logger.warning("PINECONE_API_KEY not set → RAG disabled")
            return

        if not settings.PINECONE_ENVIRONMENT:
            logger.error("PINECONE_ENVIRONMENT must be set (e.g. us-east-1)")
            return

        try:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
                logger.info(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
                pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=self.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT,
                    ),
                )
                # Wait for index to be ready
                time.sleep(30)
                logger.info("Pinecone index created and ready")

            self.index = pc.Index(settings.PINECONE_INDEX_NAME)
            
            # Test the index connection
            try:
                index_stats = self.index.describe_index_stats()
                logger.info(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' ready. Total vectors: {index_stats.get('total_vector_count', 0)}")
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}", exc_info=True)
            self.index = None

    # =====================================================
    # Text Splitter - Optimized for legal documents
    # =====================================================

    def _init_text_splitter(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Smaller chunks for legal precision
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            length_function=len,
        )

    # =====================================================
    # Groq API Helper Methods
    # =====================================================

    async def _call_groq_api(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Call Groq API with proper error handling"""
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        headers = {
            "Authorization": f"Bearer {self.client['api_key']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.client["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000
        }

        try:
            async with httpx.AsyncClient(timeout=self.client["timeout"]) as client:
                response = await client.post(
                    f"{self.client['base_url']}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"Groq API error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            logger.error("Groq API timeout")
            raise RuntimeError("AI service timeout. Please try again.")
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

    # =====================================================
    # Document Analysis (Groq)
    # =====================================================

    async def analyze_document(self, text: str) -> Dict[str, Any]:
        if not self.client or not text.strip():
            return self._fallback_analysis()

        start = time.time()

        # Truncate text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text
        
        system_prompt = """You are a legal document analysis expert. Analyze the legal document and return STRICT JSON format with the following structure:
{
  "summary": "string",
  "key_points": ["string"],
  "clauses": [],
  "risks": [],
  "recommendations": []
}

Return ONLY the JSON object, no additional text."""

        user_prompt = f"Document to analyze:\n{truncated_text}"

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response_text = await self._call_groq_api(messages, temperature=0.1)
            
            # Clean the response
            raw = response_text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```json|```$", "", raw).strip()

            data = json.loads(raw)

            for key in ["summary", "key_points", "clauses", "risks", "recommendations"]:
                data.setdefault(key, [] if key != "summary" else "")

            data["processing_time"] = time.time() - start
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Groq response: {e}")
            logger.error(f"Raw response: {raw[:500] if 'raw' in locals() else 'No response'}")
            fallback = self._fallback_analysis()
            fallback["processing_time"] = time.time() - start
            fallback["error"] = f"AI response parsing failed: {str(e)}"
            return fallback
        except Exception as e:
            logger.error(f"Groq analysis failed: {e}")
            fallback = self._fallback_analysis()
            fallback["processing_time"] = time.time() - start
            fallback["error"] = str(e)
            return fallback

    def _fallback_analysis(self) -> Dict[str, Any]:
        return {
            "summary": "Analysis unavailable. Manual legal review required.",
            "key_points": [],
            "clauses": [],
            "risks": [],
            "recommendations": ["Manual legal review required"],
            "processing_time": 0.0,
        }

    # =====================================================
    # Store Document Embeddings - FIXED with better handling
    # =====================================================

    async def store_document_embeddings(
        self,
        document_id: int,
        text: str,
        metadata: Dict[str, Any],
    ) -> bool:

        if not self.index:
            logger.error("Pinecone index is not available - embeddings will not be stored")
            return False

        if not text or not text.strip():
            logger.warning("Empty document text → skipping embeddings")
            return False

        try:
            logger.info(f"Creating embeddings for document {document_id}")
            logger.info(f"Text length: {len(text)} characters")
            
            # Split text into chunks
            chunks = self.splitter.split_text(text)
            logger.info(f"Created {len(chunks)} text chunks")
            
            if len(chunks) == 0:
                logger.error("No chunks created from text")
                return False
            
            # Log first few chunks for debugging
            for i, chunk in enumerate(chunks[:5]):
                logger.info(f"Chunk {i} (first 150 chars): {chunk[:150]}...")
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            vectors = self._embed(chunks)
            logger.info(f"Generated {len(vectors)} embeddings")
            
            if len(vectors) == 0:
                logger.error("Failed to generate any embeddings")
                return False
            
            if len(vectors) != len(chunks):
                logger.warning(f"Number of vectors ({len(vectors)}) doesn't match number of chunks ({len(chunks)})")
                # Use the smaller count
                min_len = min(len(vectors), len(chunks))
                vectors = vectors[:min_len]
                chunks = chunks[:min_len]
            
            # Prepare Pinecone vectors
            payload = []
            successful_vectors = 0
            
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                try:
                    # Clean chunk text for metadata
                    clean_chunk = chunk.replace('\n', ' ').strip()
                    
                    payload.append({
                        "id": f"doc-{document_id}-chunk-{i}",
                        "values": vec,
                        "metadata": {
                            "document_id": document_id,
                            "chunk_index": i,
                            "text": clean_chunk[:800],  # Truncate for metadata
                            "chunk_length": len(chunk),
                            **metadata,
                        },
                    })
                    successful_vectors += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare chunk {i}: {e}")
                    continue
            
            if successful_vectors == 0:
                logger.error("No valid vectors to upload")
                return False
            
            # Upload to Pinecone
            logger.info(f"Uploading {successful_vectors} vectors to Pinecone...")
            try:
                # Upload in smaller batches for reliability
                batch_size = 50
                for i in range(0, len(payload), batch_size):
                    batch = payload[i:i + batch_size]
                    response = self.index.upsert(vectors=batch)
                    logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(payload)+batch_size-1)//batch_size}")
                
                # Verify upload
                time.sleep(2)  # Wait for indexing
                
                # Check if vectors were stored by querying for this document
                test_query = self.index.query(
                    vector=vectors[0] if vectors else [0]*self.embedding_dim,
                    top_k=1,
                    filter={"document_id": {"$eq": document_id}},
                    include_metadata=False
                )
                
                if test_query.get('matches'):
                    logger.info(f"Successfully stored {successful_vectors} embeddings for document {document_id}")
                    logger.info(f"Test query returned {len(test_query['matches'])} matches")
                    return True
                else:
                    logger.error("Test query returned no matches - embeddings may not be stored")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to upload to Pinecone: {e}", exc_info=True)
                return False

        except Exception as e:
            logger.error(f"Embedding storage failed: {str(e)}", exc_info=True)
            return False

    # =====================================================
    # Delete Embeddings
    # =====================================================

    async def delete_document_embeddings(self, document_id: int) -> bool:
        if not self.index:
            return False

        try:
            logger.info(f"Deleting embeddings for document {document_id}")
            # Try different filter formats
            try:
                self.index.delete(filter={"document_id": document_id})
            except:
                self.index.delete(filter={"document_id": {"$eq": document_id}})
            
            logger.info(f"Deleted embeddings for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    # =====================================================
    # RAG Chat with Groq - IMPROVED with better retrieval
    # =====================================================

    async def chat_with_document(
        self,
        document_id: int,
        question: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:

        start = time.time()

        try:
            logger.info(f"Chat request - Document: {document_id}, Question: '{question}'")
            
            # Check if Pinecone is available
            if not self.index:
                logger.error("Pinecone index not available")
                return {
                    "answer": "Vector database is not configured. Please check the AI configuration.",
                    "sources": [],
                    "response_time": time.time() - start,
                }

            # Check if Groq is available
            if not self.client:
                logger.error("Groq client not available")
                return {
                    "answer": "AI model is not configured. Please check the AI configuration.",
                    "sources": [],
                    "response_time": time.time() - start,
                }

            # Generate query embedding
            logger.info("Generating query embedding...")
            query_vectors = self._embed([question])
            if not query_vectors:
                logger.error("Failed to generate query embedding")
                return {
                    "answer": "Failed to process your question. Please try again.",
                    "sources": [],
                    "response_time": time.time() - start,
                }
            
            query_vec = query_vectors[0]
            
            # Query Pinecone for relevant chunks
            logger.info("Querying Pinecone for relevant chunks...")
            try:
                results = self.index.query(
                    vector=query_vec,
                    top_k=8,  # Get more results
                    filter={"document_id": {"$eq": document_id}},
                    include_metadata=True,
                    include_values=False,
                )
                
                matches = results.get('matches', [])
                logger.info(f"Found {len(matches)} matches from Pinecone")
                
                # Debug: Log the matches
                for i, match in enumerate(matches[:5]):  # Log first 5
                    score = match.get('score', 0)
                    meta = match.get('metadata', {})
                    chunk_preview = meta.get('text', '')[:100] if meta.get('text') else 'No text'
                    logger.info(f"Match {i+1}: Score={score:.4f}, Chunk: '{chunk_preview}...'")
                
            except Exception as e:
                logger.error(f"Pinecone query failed: {e}")
                # Try without filter as fallback
                try:
                    results = self.index.query(
                        vector=query_vec,
                        top_k=5,
                        include_metadata=True,
                        include_values=False,
                    )
                    matches = results.get('matches', [])
                    logger.info(f"Found {len(matches)} matches (no filter)")
                except Exception as e2:
                    logger.error(f"Fallback Pinecone query also failed: {e2}")
                    matches = []

            # Build context from matches
            context_chunks = []
            sources = []
            
            for match in matches:
                score = match.get('score', 0)
                metadata = match.get('metadata', {})
                chunk_text = metadata.get('text', '')
                
                # Only use chunks with reasonable similarity score
                if score > 0.3 and chunk_text:
                    context_chunks.append(chunk_text)
                    sources.append({
                        "chunk_index": metadata.get('chunk_index', 0),
                        "score": float(score),
                    })
            
            context = "\n\n".join(context_chunks)
            logger.info(f"Built context from {len(context_chunks)} chunks, total length: {len(context)} chars")
            
            if not context or len(context.strip()) < 50:
                logger.warning("Context is too short or empty")
                return {
                    "answer": "I cannot find sufficient information in the document to answer this question. The document may not contain relevant information or the embeddings may need to be regenerated.",
                    "sources": [],
                    "response_time": time.time() - start,
                }

            # Prepare the chat prompt with context
            system_prompt = """You are a helpful legal document assistant. Answer questions based STRICTLY on the provided context from a legal document.

IMPORTANT RULES:
1. Use ONLY the information in the provided context
2. If the context has the answer, provide it clearly
3. If you cannot find the answer in the context, say: "Based on the provided context, I cannot find specific information about [topic]."
4. Do NOT make up any information
5. Keep answers concise but complete"""

            user_prompt = f"""Here is context from a legal document:

{context}

Based ONLY on the context above, answer this question:

{question}

Remember:
- Answer using ONLY the information from the context
- If the answer isn't in the context, say so
- Be specific and accurate"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call Groq API
            logger.info("Calling Groq API...")
            response_text = await self._call_groq_api(messages, temperature=temperature)
            
            response_time = time.time() - start
            logger.info(f"Generated response in {response_time:.2f} seconds")
            logger.info(f"Response preview: {response_text[:200]}...")

            return {
                "answer": response_text,
                "sources": sources[:5],  # Return top 5 sources
                "response_time": response_time,
            }

        except Exception as e:
            logger.error(f"Chat failed: {str(e)}", exc_info=True)
            return {
                "answer": "An error occurred while processing your request. Please try again.",
                "sources": [],
                "response_time": time.time() - start,
            }


# =========================================================
# Singleton Instance
# =========================================================

ai_processor = AIProcessor()