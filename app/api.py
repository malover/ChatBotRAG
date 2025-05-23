import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Add root directory to path for imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.rag_system import MicroplasticsRAG, RAGResponse, SearchResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system: Optional[MicroplasticsRAG] = None


# Pydantic models for API
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Research question about microplastics")
    top_k: int = Field(default=7, ge=1, le=20, description="Number of relevant sources to retrieve")
    include_sources: bool = Field(default=True, description="Whether to include source information")


class SourceInfo(BaseModel):
    id: str
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    title: str
    authors: List[str]
    journal: str
    url: str
    content_type: str
    chunk_index: int
    text_preview: str = Field(..., description="First 200 characters of the source text")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SystemInfoResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
    collection_info: Dict[str, Any]
    gpu_info: Dict[str, Any]
    configuration: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    status: str
    uptime: float
    system_ready: bool
    components: Dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.now)


# Application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global rag_system

    # Startup
    logger.info("🚀 Starting Microplastics Research API...")

    try:
        # Initialize RAG system
        logger.info("🔄 Initializing RAG system...")
        rag_system = MicroplasticsRAG()

        # Test the system
        system_info = rag_system.get_system_info()
        logger.info(f"✅ RAG system initialized successfully!")
        logger.info(f"📚 Knowledge base: {system_info.get('collection', {}).get('points', 0)} documents")
        logger.info(
            f"🧠 Models: {system_info.get('embedding_model', 'Unknown')} + {system_info.get('llm_model', 'Unknown')}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        logger.error("🔧 Make sure:")
        logger.error("   1. Ollama is running: ollama serve")
        logger.error("   2. Model is installed: ollama pull llama3:8b")
        logger.error("   3. Qdrant has data: python index_to_qdrant.py")
        rag_system = None

    yield

    # Shutdown
    logger.info("👋 Shutting down Microplastics Research API...")


# Create FastAPI app
app = FastAPI(
    title="Microplastics Research API",
    description="AI-powered research assistant for microplastic pollution in marine ecosystems",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store startup time for uptime calculation
startup_time = datetime.now()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Microplastics Research API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - startup_time).total_seconds()

    # Check system components
    components = {
        "rag_system": "ready" if rag_system is not None else "not_initialized",
        "api": "ready"
    }

    # Additional checks if RAG system is available
    if rag_system:
        try:
            system_info = rag_system.get_system_info()
            components["vector_db"] = "ready" if system_info.get('collection', {}).get('points', 0) > 0 else "empty"
            components["embedding_model"] = "ready"
            components["llm"] = "ready"
        except Exception as e:
            components["vector_db"] = f"error: {str(e)}"
            components["embedding_model"] = "error"
            components["llm"] = "error"

    system_ready = all(status.startswith("ready") for status in components.values())

    return HealthResponse(
        status="healthy" if system_ready else "degraded",
        uptime=uptime,
        system_ready=system_ready,
        components=components
    )


@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get detailed system information."""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Check server logs for details."
        )

    try:
        system_info = rag_system.get_system_info()

        return SystemInfoResponse(
            status="ready",
            embedding_model=system_info.get('embedding_model', 'Unknown'),
            llm_model=system_info.get('llm_model', 'Unknown'),
            collection_info=system_info.get('collection', {}),
            gpu_info={
                "available": system_info.get('gpu_available', False),
                "name": system_info.get('gpu_name', 'N/A')
            },
            configuration=system_info.get('configuration', {})
        )

    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving system information: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_research(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a research query and return AI-generated response with sources."""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check system health endpoint for details."
        )

    try:
        logger.info(f"Processing query: {request.question[:100]}...")

        # Process the query
        rag_response: RAGResponse = rag_system.query(
            question=request.question,
            top_k=request.top_k
        )

        # Convert sources to API format
        sources = []
        if request.include_sources:
            for source in rag_response.sources:
                sources.append(SourceInfo(
                    id=source.id,
                    score=source.score,
                    title=source.title,
                    authors=source.authors,
                    journal=source.journal,
                    url=source.url,
                    content_type=source.content_type,
                    chunk_index=source.chunk_index,
                    text_preview=source.text[:200] + "..." if len(source.text) > 200 else source.text
                ))

        # Prepare metadata
        metadata = {
            "sources_found": len(rag_response.sources),
            "query_length": len(request.question),
            "answer_length": len(rag_response.answer)
        }

        # Performance metrics
        performance = {
            "retrieval_time": rag_response.retrieval_time,
            "generation_time": rag_response.generation_time,
            "total_time": rag_response.total_time
        }

        logger.info(f"Query completed in {rag_response.total_time:.2f}s")

        return QueryResponse(
            question=rag_response.question,
            answer=rag_response.answer,
            sources=sources,
            metadata=metadata,
            performance=performance
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/sources/search", response_model=List[SourceInfo])
async def search_sources(
        query: str,
        limit: int = 10
):
    """Search for relevant sources without generating an answer."""
    # Validate parameters manually
    if len(query.strip()) < 3:
        raise HTTPException(status_code=422, detail="Query must be at least 3 characters long")

    if limit < 1 or limit > 50:
        raise HTTPException(status_code=422, detail="Limit must be between 1 and 50")

    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )

    try:
        logger.info(f"Searching sources for: {query[:100]}...")

        # Use the retriever directly
        search_results = rag_system.retriever.search(query, top_k=limit)

        # Convert to API format
        sources = []
        for result in search_results:
            sources.append(SourceInfo(
                id=result.id,
                score=result.score,
                title=result.title,
                authors=result.authors,
                journal=result.journal,
                url=result.url,
                content_type=result.content_type,
                chunk_index=result.chunk_index,
                text_preview=result.text[:200] + "..." if len(result.text) > 200 else result.text
            ))

        logger.info(f"Found {len(sources)} relevant sources")
        return sources

    except Exception as e:
        logger.error(f"Error searching sources: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching sources: {str(e)}")


@app.get("/examples", response_model=Dict[str, List[str]])
async def get_example_queries():
    """Get example queries to help users understand what they can ask."""
    return {
        "general_questions": [
            "What are microplastics and where do they come from?",
            "How do microplastics affect marine life?",
            "What are the main sources of microplastic pollution?",
            "How can we reduce microplastic pollution in oceans?"
        ],
        "technical_questions": [
            "What methods are used to detect microplastics in seawater?",
            "How do microplastics interact with marine food webs?",
            "What are the bioaccumulation patterns of microplastics?",
            "Which analytical techniques are most effective for microplastic characterization?"
        ],
        "environmental_questions": [
            "How are microplastics distributed in marine environments?",
            "What factors influence microplastic transport in oceans?",
            "How do microplastics affect marine ecosystem health?",
            "What are the long-term environmental impacts of microplastic pollution?"
        ],
        "solutions_questions": [
            "What are the most promising solutions to reduce microplastic pollution?",
            "How effective are current microplastic removal technologies?",
            "What policies could help address microplastic pollution?",
            "How can we prevent microplastics from entering marine environments?"
        ]
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please check the server logs.",
            "timestamp": datetime.now().isoformat()
        }
    )


def main():
    """Run the FastAPI server."""
    logger.info("🌊 Starting Microplastics Research API Server...")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
