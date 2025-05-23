import json
import logging
import requests
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from scripts.config_rtx4080 import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_PARAMETERS,
    TOP_K_RESULTS, SIMILARITY_THRESHOLD, SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE,
    get_gpu_info
)

# Import required libraries
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Load configuration from parent directory
import sys
from pathlib import Path

# Add root directory to path to import config
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result from vector database."""
    id: str
    score: float
    text: str
    title: str
    authors: List[str]
    journal: str
    url: str
    content_type: str
    chunk_index: int


@dataclass
class RAGResponse:
    """Complete RAG response with sources."""
    question: str
    answer: str
    sources: List[SearchResult]
    retrieval_time: float
    generation_time: float
    total_time: float


class VectorRetriever:
    """Handles vector similarity search in Qdrant."""

    def __init__(self):
        logger.info("Initializing Vector Retriever...")

        # Initialize Qdrant client
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = QDRANT_COLLECTION_NAME

        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

        # Verify collection exists
        self._verify_collection()

        logger.info("✅ Vector Retriever initialized successfully")

    def _verify_collection(self):
        """Verify that the collection exists and has data."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            point_count = collection_info.points_count

            if point_count == 0:
                logger.warning(f"Collection '{self.collection_name}' is empty!")
                logger.warning("Run index_to_qdrant.py to populate the database")
            else:
                logger.info(f"Collection '{self.collection_name}' has {point_count} points")

        except Exception as e:
            logger.error(f"Error accessing collection '{self.collection_name}': {e}")
            raise

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[SearchResult]:
        """Search for relevant documents using vector similarity."""
        logger.debug(f"Searching for: '{query}' (top_k={top_k})")

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding[0].tolist(),
                limit=top_k,
                score_threshold=SIMILARITY_THRESHOLD
            )

            # Convert to SearchResult objects
            results = []
            for result in search_results:
                payload = result.payload
                search_result = SearchResult(
                    id=str(result.id),
                    score=float(result.score),
                    text=payload.get('text', ''),
                    title=payload.get('title', 'Unknown Title'),
                    authors=payload.get('authors', []),
                    journal=payload.get('journal', ''),
                    url=payload.get('url', ''),
                    content_type=payload.get('content_type', 'general_content'),
                    chunk_index=payload.get('chunk_index', 0)
                )
                results.append(search_result)

            logger.debug(f"Found {len(results)} relevant documents")
            return results

        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []


class LlamaGenerator:
    """Handles text generation using Ollama."""

    def __init__(self):
        logger.info("Initializing Llama Generator...")

        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.generation_params = OLLAMA_PARAMETERS.copy()

        # Test Ollama connection
        self._test_connection()

        logger.info("✅ Llama Generator initialized successfully")

    def _test_connection(self):
        """Test connection to Ollama service."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]

                if self.model in model_names:
                    logger.info(f"✅ Model '{self.model}' is available")
                else:
                    logger.warning(f"⚠️  Model '{self.model}' not found in Ollama")
                    logger.warning(f"Available models: {model_names}")
                    logger.warning(f"Run: ollama pull {self.model}")
            else:
                logger.error(f"Ollama API returned status {response.status_code}")

        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to Ollama service")
            logger.error("Make sure Ollama is running: ollama serve")
            raise
        except Exception as e:
            logger.error(f"Error testing Ollama connection: {e}")
            raise

    def generate(self, prompt: str) -> str:
        """Generate response using Ollama."""
        logger.debug(f"Generating response for prompt ({len(prompt)} chars)")

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": self.generation_params
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 minute timeout for generation
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')

                logger.debug(f"Generated {len(generated_text)} characters")
                return generated_text
            else:
                logger.error(f"Ollama generation failed: {response.status_code}")
                return f"Error: Generation failed with status {response.status_code}"

        except requests.exceptions.Timeout:
            logger.error("Generation timed out")
            return "Error: Response generation timed out. Try a simpler question."
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error: {str(e)}"


class MicroplasticsRAG:
    """Main RAG system combining retrieval and generation."""

    def __init__(self):
        logger.info("Initializing Microplastics RAG System...")

        # Initialize components
        self.retriever = VectorRetriever()
        self.generator = LlamaGenerator()

        logger.info("🎉 Microplastics RAG System ready!")

    def query(self, question: str, top_k: int = TOP_K_RESULTS) -> RAGResponse:
        """Process a complete RAG query."""
        import time

        logger.info(f"Processing query: {question}")
        start_time = time.time()

        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        search_results = self.retriever.search(question, top_k=top_k)
        retrieval_time = time.time() - retrieval_start

        if not search_results:
            logger.warning("No relevant documents found")
            return RAGResponse(
                question=question,
                answer="I couldn't find any relevant information about your question in the research database. Please try rephrasing your question or ask about microplastic pollution, marine ecosystems, or related topics.",
                sources=[],
                retrieval_time=retrieval_time,
                generation_time=0.0,
                total_time=time.time() - start_time
            )

        # Step 2: Format context
        context = self._format_context(search_results)

        # Step 3: Generate response
        generation_start = time.time()
        prompt = self._build_prompt(question, context)
        answer = self.generator.generate(prompt)
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        logger.info(
            f"Query completed in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")

        return RAGResponse(
            question=question,
            answer=answer,
            sources=search_results,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time
        )

    def _format_context(self, search_results: List[SearchResult]) -> str:
        """Format search results into context for the LLM."""
        context_parts = []

        for i, result in enumerate(search_results, 1):
            # Create source citation
            source_info = f"Source {i}: {result.title}"
            if result.authors:
                authors_str = ", ".join(result.authors[:3])  # First 3 authors
                if len(result.authors) > 3:
                    authors_str += " et al."
                source_info += f" by {authors_str}"

            if result.journal:
                source_info += f" ({result.journal})"

            # Add content with source
            context_part = f"{source_info}\nContent: {result.text}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the complete prompt for the LLM."""
        # Combine system prompt with RAG template
        full_prompt = f"{SYSTEM_PROMPT}\n\n{RAG_PROMPT_TEMPLATE}"

        # Fill in the template
        return full_prompt.format(context=context, question=question)

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system."""
        try:
            # Collection info
            collection_info = self.retriever.client.get_collection(self.retriever.collection_name)

            # GPU info
            gpu_info = get_gpu_info()

            return {
                "embedding_model": EMBEDDING_MODEL,
                "llm_model": OLLAMA_MODEL,
                "collection": {
                    "name": collection_info.name,
                    "points": collection_info.points_count,
                    "vectors": collection_info.vectors_count
                },
                "gpu_available": gpu_info.get("available", False),
                "gpu_name": gpu_info.get("name", "N/A"),
                "configuration": {
                    "top_k_results": TOP_K_RESULTS,
                    "similarity_threshold": SIMILARITY_THRESHOLD
                }
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}


def main():
    """Test the RAG system with sample queries."""
    print("🧬 Microplastics Research RAG System")
    print("=" * 50)

    try:
        # Initialize RAG system
        rag_system = MicroplasticsRAG()

        # Show system info
        system_info = rag_system.get_system_info()
        print(f"📊 System Information:")
        print(f"   Embedding Model: {system_info.get('embedding_model', 'Unknown')}")
        print(f"   LLM Model: {system_info.get('llm_model', 'Unknown')}")
        print(f"   Collection: {system_info.get('collection', {}).get('name', 'Unknown')}")
        print(f"   Documents: {system_info.get('collection', {}).get('points', 0)}")
        print(f"   GPU: {system_info.get('gpu_name', 'N/A')}")

        # Test with sample queries
        test_queries = [
            "What are the main sources of microplastics in marine environments?",
            "How do microplastics affect marine organisms?",
            "What methods are used to detect microplastics in seawater?"
        ]

        print(f"\n🧪 Testing with sample queries...")

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'=' * 20} Query {i} {'=' * 20}")
            print(f"❓ Question: {query}")

            # Process query
            response = rag_system.query(query)

            print(f"\n💡 Answer:")
            print(response.answer)

            print(f"\n📚 Sources ({len(response.sources)}):")
            for j, source in enumerate(response.sources, 1):
                print(f"   {j}. {source.title} (relevance: {source.score:.3f})")
                if source.journal:
                    print(f"      Journal: {source.journal}")

            print(f"\n⏱️  Performance:")
            print(f"   Retrieval: {response.retrieval_time:.2f}s")
            print(f"   Generation: {response.generation_time:.2f}s")
            print(f"   Total: {response.total_time:.2f}s")

        print(f"\n🎉 RAG system test completed!")
        print(f"💡 The system is ready for interactive use")

    except Exception as e:
        logger.error(f"RAG system test failed: {e}")
        print(f"❌ Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check Ollama is running: ollama serve")
        print(f"   2. Verify model is installed: ollama pull {OLLAMA_MODEL}")
        print(f"   3. Check Qdrant connection: python test_qdrant_connection.py")


if __name__ == "__main__":
    main()