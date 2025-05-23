import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# Load configuration from .env
from config_rtx4080 import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDING_BATCH_SIZE, QDRANT_BATCH_SIZE,
    USE_GPU, USE_HALF_PRECISION, validate_config, get_gpu_info
)

# Embedding model
from sentence_transformers import SentenceTransformer

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("../data")
PROCESSED_DIR = DATA_DIR / "processed"


class EmbeddingGenerator:
    """Generates embeddings for text chunks using RTX 4080."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")

        # Configure for GPU usage
        import torch
        self.device = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
        logger.info(f"Using device: {self.device}")

        if self.device == "cuda":
            # Show GPU info
            gpu_info = get_gpu_info()
            if gpu_info["available"]:
                logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)

        # Enable mixed precision for RTX 4080
        if self.device == "cuda" and USE_HALF_PRECISION:
            self.model.half()  # Use FP16 for faster inference and less VRAM
            logger.info("Enabled FP16 mixed precision")

        logger.info(f"Embedding model loaded successfully")
        logger.info(f"Model max sequence length: {self.model.max_seq_length}")
        logger.info(f"Model output dimensions: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Generate embeddings for a list of texts with GPU acceleration."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        logger.info(f"Batch size: {batch_size}, Model: {EMBEDDING_MODEL}")

        embeddings = []
        current_batch_size = batch_size

        # Process in batches to manage GPU memory
        for i in tqdm(range(0, len(texts), current_batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + current_batch_size]

            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Normalize for cosine similarity
                    batch_size=current_batch_size
                )
                embeddings.append(batch_embeddings)

                # Log progress every 10 batches
                if (i // current_batch_size + 1) % 10 == 0:
                    logger.debug(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU OOM at batch {i // current_batch_size + 1}, reducing batch size")

                    # Clear GPU cache
                    if self.device == "cuda":
                        import torch
                        torch.cuda.empty_cache()

                    # Retry with smaller batch
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.info(f"Retrying with batch size: {current_batch_size}")

                    batch_embeddings = self.model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=current_batch_size
                    )
                    embeddings.append(batch_embeddings)
                else:
                    logger.error(f"Unexpected CUDA error: {e}")
                    raise e

        # Combine all embeddings
        all_embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")

        return all_embeddings

    def test_embedding(self, test_text: str = "microplastic pollution in marine environment") -> np.ndarray:
        """Test embedding generation with a sample text."""
        logger.info(f"Testing embedding generation...")
        embedding = self.model.encode([test_text], normalize_embeddings=True)
        logger.info(f"Test embedding shape: {embedding.shape}")
        return embedding[0]


class QdrantIndexer:
    """Indexes documents into Qdrant Cloud."""

    def __init__(self, url: str = QDRANT_URL, api_key: str = QDRANT_API_KEY):
        self.collection_name = QDRANT_COLLECTION_NAME

        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {url}")
        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info("✅ Connected to Qdrant successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise

        # Setup collection
        self._setup_collection()

    def _setup_collection(self):
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                # Get collection info
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"Existing collection has {collection_info.points_count} points")

                # Check if dimensions match
                existing_dim = collection_info.config.params.vectors.size
                if existing_dim != EMBEDDING_DIMENSION:
                    logger.warning(f"Dimension mismatch! Existing: {existing_dim}, New: {EMBEDDING_DIMENSION}")
                    logger.warning("Consider deleting the collection or using a different collection name")
            else:
                logger.info(f"Creating collection '{self.collection_name}'")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info("✅ Collection created successfully")

        except Exception as e:
            logger.error(f"❌ Error setting up collection: {e}")
            raise

    def index_documents(self, chunks: List[Dict], embeddings: np.ndarray, batch_size: int = QDRANT_BATCH_SIZE):
        """Index documents with embeddings into Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        logger.info(f"Indexing {len(chunks)} documents into Qdrant")
        logger.info(f"Upload batch size: {batch_size}")

        # Prepare points for insertion
        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create comprehensive payload with metadata
            payload = {
                # Core content
                'text': chunk['text'],
                'chunk_id': chunk['id'],

                # Paper metadata
                'paper_id': chunk['metadata']['paper_id'],
                'title': chunk['metadata']['title'],
                'authors': chunk['metadata']['authors'],
                'journal': chunk['metadata']['journal'],
                'publication_date': chunk['metadata']['publication_date'],
                'url': chunk['metadata']['url'],
                'keywords': chunk['metadata']['keywords'],
                'domain': chunk['metadata']['domain'],

                # Chunk metadata
                'chunk_index': chunk['metadata']['chunk_index'],
                'total_chunks': chunk['metadata']['total_chunks'],
                'chunk_size': chunk['metadata']['chunk_size'],
                'content_type': chunk['metadata']['content_type'],
                'has_title': chunk['metadata']['has_title'],
                'has_abstract': chunk['metadata']['has_abstract']
            }

            # Create point with unique UUID
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

            # Log progress for large datasets
            if (i + 1) % 1000 == 0:
                logger.debug(f"Prepared {i + 1}/{len(chunks)} points")

        logger.info(f"Prepared {len(points)} points for upload")

        # Upload in batches
        total_uploaded = 0
        failed_batches = 0

        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch_points = points[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                result = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                total_uploaded += len(batch_points)

                logger.debug(f"Batch {batch_num}: uploaded {len(batch_points)} points")

            except Exception as e:
                failed_batches += 1
                logger.error(f"❌ Error uploading batch {batch_num}: {e}")

                # Try to continue with smaller batches
                if len(batch_points) > 1:
                    logger.info(f"Retrying batch {batch_num} with smaller size...")
                    smaller_batch_size = len(batch_points) // 2

                    for j in range(0, len(batch_points), smaller_batch_size):
                        smaller_batch = batch_points[j:j + smaller_batch_size]
                        try:
                            self.client.upsert(
                                collection_name=self.collection_name,
                                points=smaller_batch
                            )
                            total_uploaded += len(smaller_batch)
                        except Exception as e2:
                            logger.error(f"❌ Failed even with smaller batch: {e2}")
                            break
                else:
                    logger.error(f"❌ Failed to upload single point in batch {batch_num}")

        logger.info(f"Upload complete: {total_uploaded}/{len(points)} points uploaded")
        if failed_batches > 0:
            logger.warning(f"⚠️  {failed_batches} batches had errors")

        return total_uploaded

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': collection_info.name,
                'points_count': collection_info.points_count,
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status,
                'config': {
                    'distance': collection_info.config.params.vectors.distance,
                    'size': collection_info.config.params.vectors.size
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def search_similar(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict]:
        """Search for similar documents."""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit
            )

            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    'title': result.payload.get('title', ''),
                    'content_type': result.payload.get('content_type', ''),
                    'text_preview': result.payload.get('text', '')[:200] + '...',
                    'metadata': result.payload
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []


def load_processed_chunks() -> List[Dict]:
    """Load processed chunks from file."""
    chunks_file = PROCESSED_DIR / "processed_chunks.json"

    if not chunks_file.exists():
        logger.error(f"Processed chunks file not found: {chunks_file}")
        logger.info("Run preprocess_documents.py first")
        return []

    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} processed chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error loading processed chunks: {e}")
        return []


def main():
    """Main indexing function."""
    print("☁️ Microplastics Research - Qdrant Indexing")
    print("=" * 60)

    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        print("❌ Configuration Errors:")
        for error in config_errors:
            print(f"   - {error}")
        print("\n💡 Please fix your .env file first")
        return

    # Show configuration
    gpu_info = get_gpu_info()
    print(f"⚙️  Configuration:")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print(f"   Dimensions: {EMBEDDING_DIMENSION}")
    print(f"   Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
    print(f"   Qdrant Batch Size: {QDRANT_BATCH_SIZE}")
    if gpu_info["available"]:
        print(f"   GPU: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")
        print(f"   Half Precision: {USE_HALF_PRECISION}")
    else:
        print("   Device: CPU (no GPU detected)")

    try:
        # Step 1: Load processed chunks
        print(f"\n📁 Step 1: Loading processed chunks...")
        chunks = load_processed_chunks()

        print(f"✅ Loaded {len(chunks)} chunks")

        # Step 2: Initialize embedding generator
        print(f"\n🧠 Step 2: Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator()

        # Test embedding generation
        test_embedding = embedding_generator.test_embedding()
        print(f"✅ Embedding generator ready (test embedding: {test_embedding.shape})")

        # Step 3: Generate embeddings for all chunks
        print(f"\n⚡ Step 3: Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]

        # Show some statistics about the texts
        text_lengths = [len(text) for text in texts]
        print(f"   Text statistics:")
        print(f"     Min length: {min(text_lengths)} chars")
        print(f"     Max length: {max(text_lengths)} chars")
        print(f"     Average: {sum(text_lengths) / len(text_lengths):.0f} chars")

        embeddings = embedding_generator.generate_embeddings(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")

        # Step 4: Initialize Qdrant indexer
        print(f"\n☁️ Step 4: Connecting to Qdrant...")
        indexer = QdrantIndexer()

        # Check existing collection
        collection_info = indexer.get_collection_info()
        if collection_info:
            print(f"✅ Connected to collection: {collection_info['name']}")
            print(f"   Existing points: {collection_info['points_count']}")
            print(f"   Vector dimensions: {collection_info['config']['size']}")

        # Step 5: Index documents
        print(f"\n📤 Step 5: Uploading to Qdrant...")
        uploaded_count = indexer.index_documents(chunks, embeddings)

        # Step 6: Verify indexing
        print(f"\n🔍 Step 6: Verifying upload...")
        final_collection_info = indexer.get_collection_info()

        if final_collection_info:
            print(f"✅ Upload verification:")
            print(f"   Total points in collection: {final_collection_info['points_count']}")
            print(f"   Upload success rate: {uploaded_count}/{len(chunks)} ({uploaded_count / len(chunks) * 100:.1f}%)")

        # Step 7: Test search functionality
        print(f"\n🧪 Step 7: Testing search functionality...")
        test_queries = [
            "microplastic pollution marine environment",
            "sources of microplastics in oceans",
            "effects on marine life and organisms"
        ]

        for query in test_queries:
            print(f"\n   Testing query: '{query}'")
            query_embedding = embedding_generator.model.encode([query], normalize_embeddings=True)
            results = indexer.search_similar(query_embedding[0], limit=3)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"     {i}. {result['title'][:50]}... (score: {result['score']:.3f})")
                    print(f"        Type: {result['content_type']}")
            else:
                print("     ❌ No results found")

        print(f"\n" + "=" * 60)
        print("✅ Indexing completed successfully!")

        # Final summary
        unique_papers = len(set(chunk['metadata']['paper_id'] for chunk in chunks))
        print(f"\n📊 Final Summary:")
        print(f"   📄 Papers indexed: {unique_papers}")
        print(f"   🧩 Text chunks: {len(chunks)}")
        print(f"   🧠 Embeddings: {len(embeddings)} ({EMBEDDING_DIMENSION}D)")
        print(f"   ☁️  Qdrant points: {final_collection_info.get('points_count', 0)}")
        print(f"   🎮 GPU acceleration: {gpu_info['available'] and USE_GPU}")

        # Show content distribution
        content_types = {}
        for chunk in chunks:
            content_type = chunk['metadata']['content_type']
            content_types[content_type] = content_types.get(content_type, 0) + 1

        print(f"\n📋 Content Type Distribution:")
        for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(chunks)) * 100
            print(f"   {content_type}: {count} chunks ({percentage:.1f}%)")

        print(f"\n🎉 Vector database is ready for RAG queries!")
        print(f"   Collection: {QDRANT_COLLECTION_NAME}")
        print(f"   Dimensions: {EMBEDDING_DIMENSION}")
        print(f"   Distance: Cosine similarity")

        # Save indexing results
        results_file = PROCESSED_DIR / "indexing_results.json"
        indexing_results = {
            'timestamp': str(datetime.now()),
            'configuration': {
                'embedding_model': EMBEDDING_MODEL,
                'embedding_dimension': EMBEDDING_DIMENSION,
                'embedding_batch_size': EMBEDDING_BATCH_SIZE,
                'qdrant_batch_size': QDRANT_BATCH_SIZE,
                'use_gpu': USE_GPU,
                'use_half_precision': USE_HALF_PRECISION
            },
            'results': {
                'chunks_processed': len(chunks),
                'embeddings_generated': len(embeddings),
                'points_uploaded': uploaded_count,
                'final_collection_size': final_collection_info.get('points_count', 0)
            },
            'collection_info': final_collection_info
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(indexing_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"📁 Indexing results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        print(f"❌ Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check your .env file configuration")
        print(f"   2. Verify Qdrant connection: python test_qdrant_connection.py")
        print(f"   3. Ensure you have enough GPU memory")
        print(f"   4. Try reducing EMBEDDING_BATCH_SIZE in .env")


if __name__ == "__main__":
    # Import datetime here since we use it in main()
    from datetime import datetime

    main()