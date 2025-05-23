import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path('../') / '.env'
load_dotenv(dotenv_path=env_path)

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "microplastics_research")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

# Embedding Configuration (from .env)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "12"))
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "100"))

# RAG Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "7"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# GPU Configuration
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
USE_HALF_PRECISION = os.getenv("USE_HALF_PRECISION", "true").lower() == "true"
MAX_GPU_MEMORY_GB = int(os.getenv("MAX_GPU_MEMORY_GB", "14"))

# Performance Monitoring
ENABLE_GPU_MONITORING = os.getenv("ENABLE_GPU_MONITORING", "true").lower() == "true"
LOG_PERFORMANCE_METRICS = os.getenv("LOG_PERFORMANCE_METRICS", "true").lower() == "true"

# Debug Configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Prompt Templates Optimized for Llama 3
SYSTEM_PROMPT = """You are a specialized research assistant focusing on microplastic pollution in marine ecosystems. You have access to recent scientific papers and research data.

Your expertise includes:
- Sources and pathways of microplastic pollution
- Environmental distribution and fate of microplastics  
- Biological impacts on marine organisms
- Detection and analysis methods
- Mitigation and prevention strategies

Guidelines:
- Provide accurate, science-based answers grounded in the research context
- Cite specific sources when making claims
- Explain technical concepts clearly for diverse audiences
- Acknowledge limitations in the available data
- Distinguish between established facts and ongoing research questions"""

RAG_PROMPT_TEMPLATE = """Context from recent microplastics research:

{context}

Question: {question}

Based on the research context above, provide a comprehensive answer that:
1. Directly addresses the question
2. References relevant findings from the provided papers
3. Explains any technical terms for clarity
4. Notes any limitations or uncertainties in the current research

Answer:"""

# Ollama Generation Parameters (Optimized for Llama 3)
OLLAMA_PARAMETERS = {
    "temperature": 0.1,  # Low temperature for factual responses
    "top_k": 40,  # Balanced creativity
    "top_p": 0.9,  # Nucleus sampling
    "repeat_penalty": 1.1,  # Avoid repetition
    "num_predict": 1024,  # Max response length
    "stop": ["Human:", "Question:", "\n\nQuestion:"]  # Stop sequences
}


def validate_config():
    """Validate that required configuration is set."""
    errors = []

    if not QDRANT_URL or QDRANT_URL == "https://your-cluster-url.qdrant.io:6333":
        errors.append("QDRANT_URL not set or using placeholder value")

    if not QDRANT_API_KEY or QDRANT_API_KEY == "your-api-key-here":
        errors.append("QDRANT_API_KEY not set or using placeholder value")

    return errors


def get_gpu_info():
    """Get GPU information if available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return {
                "name": gpu_name,
                "memory_gb": round(gpu_memory, 1),
                "cuda_version": torch.version.cuda,
                "available": True
            }
    except ImportError:
        pass
    return {"available": False}


def print_config():
    """Print current configuration."""
    print("🔧 RTX 4080 Configuration (from .env)")
    print("=" * 50)

    # Validate configuration
    errors = validate_config()
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        print("\n💡 Please update your .env file with correct values")
        return False

    print("✅ Configuration Valid")
    print(f"\n📊 Model Configuration:")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print(f"   Embedding Dimensions: {EMBEDDING_DIMENSION}")
    print(f"   Ollama Model: {OLLAMA_MODEL}")
    print(f"   Ollama URL: {OLLAMA_BASE_URL}")

    print(f"\n📝 Processing Settings:")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"   Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
    print(f"   Qdrant Batch Size: {QDRANT_BATCH_SIZE}")

    print(f"\n🔍 RAG Settings:")
    print(f"   Top K Results: {TOP_K_RESULTS}")
    print(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}")

    gpu_info = get_gpu_info()
    print(f"\n🎮 GPU Configuration:")
    if gpu_info["available"]:
        print(f"   GPU: {gpu_info['name']}")
        print(f"   Memory: {gpu_info['memory_gb']} GB")
        print(f"   CUDA: {gpu_info['cuda_version']}")
        print(f"   Use GPU: {USE_GPU}")
        print(f"   Half Precision: {USE_HALF_PRECISION}")
        print(f"   Max GPU Memory: {MAX_GPU_MEMORY_GB} GB")
    else:
        print("   ❌ No GPU detected or PyTorch not installed")

    print(f"\n☁️ Qdrant Settings:")
    print(f"   URL: {QDRANT_URL}")
    print(f"   Collection: {QDRANT_COLLECTION_NAME}")
    print(f"   API Key: {'*' * (len(QDRANT_API_KEY) - 4) + QDRANT_API_KEY[-4:] if QDRANT_API_KEY else 'Not set'}")

    print(f"\n🐛 Debug Settings:")
    print(f"   Debug Mode: {DEBUG}")
    print(f"   Log Level: {LOG_LEVEL}")
    print(f"   GPU Monitoring: {ENABLE_GPU_MONITORING}")
    print(f"   Performance Metrics: {LOG_PERFORMANCE_METRICS}")

    return True


if __name__ == "__main__":
    success = print_config()
    if not success:
        print(f"\n📝 To fix configuration errors:")
        print(f"1. Copy .env.template to .env")
        print(f"2. Edit .env with your actual Qdrant credentials")
        print(f"3. Run this script again to validate")