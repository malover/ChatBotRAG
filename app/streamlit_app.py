import sys
import requests
import streamlit as st
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Add root directory to path for imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Page config
st.set_page_config(
    page_title="🌊 Microplastics Research Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120

# Custom CSS for better styling with dark theme support
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #64b5f6;
        padding: 1rem 0;
        border-bottom: 2px solid #424242;
        margin-bottom: 2rem;
    }

    .question-box {
        background-color: #2d2d2d;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #64b5f6;
        margin: 1rem 0;
    }

    .answer-box {
        background-color: #1b2b1b;
        color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #81c784;
        margin: 1rem 0;
    }

    .source-card {
        background-color: #2a2a2a;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #424242;
        margin: 0.5rem 0;
    }

    .metric-card {
        background-color: #2a2a2a;
        color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid #ff9800;
    }

    .error-box {
        background-color: #2d1b1b;
        color: #ffcdd2;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }

    .info-box {
        background-color: #1a237e;
        color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }

    /* Dark theme compatibility */
    .stApp {
        background-color: var(--background-color);
    }

    /* Improve text readability in dark theme */
    .question-box strong,
    .answer-box strong,
    .error-box strong,
    .info-box strong {
        color: #ffffff;
    }

    /* Make metric cards more visible */
    .metric-card strong {
        color: #ffcc02;
        font-size: 0.9rem;
    }

    /* Source card improvements */
    .source-card strong {
        color: #90caf9;
    }

    /* Better contrast for links in dark theme */
    .source-card a {
        color: #81c784 !important;
    }

    .source-card a:hover {
        color: #a5d6a7 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_api_health() -> Dict[str, Any]:
    """Check if the API is healthy and return status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_system_info() -> Dict[str, Any]:
    """Get system information from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/system/info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_example_queries() -> Dict[str, List[str]]:
    """Get example queries from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/examples", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading examples: {e}")
        return {}


def query_api(question: str, top_k: int = 7, include_sources: bool = True) -> Dict[str, Any]:
    """Query the API with a research question."""
    try:
        payload = {
            "question": question,
            "top_k": top_k,
            "include_sources": include_sources
        }

        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}: {response.text}"}

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The question might be too complex or the server is overloaded."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}


def search_sources(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for sources without generating an answer."""
    try:
        params = {"query": query, "limit": limit}
        response = requests.get(f"{API_BASE_URL}/sources/search", params=params, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.status_code}")
            return []

    except requests.exceptions.RequestException as e:
        st.error(f"Search error: {e}")
        return []


def display_health_status():
    """Display API health status in the sidebar."""
    health = check_api_health()

    if health.get("status") == "healthy":
        st.sidebar.success("🟢 API is healthy")
        if "uptime" in health:
            uptime_hours = health["uptime"] / 3600
            st.sidebar.text(f"Uptime: {uptime_hours:.1f} hours")
    elif health.get("status") == "degraded":
        st.sidebar.warning("🟡 API is degraded")
        if "components" in health:
            for component, status in health["components"].items():
                if not status.startswith("ready"):
                    st.sidebar.text(f"⚠️ {component}: {status}")
    else:
        st.sidebar.error("🔴 API is unavailable")
        if "error" in health:
            st.sidebar.text(f"Error: {health['error']}")


def display_system_info():
    """Display system information in the sidebar."""
    with st.sidebar.expander("🔧 System Information", expanded=False):
        system_info = get_system_info()

        if "error" in system_info:
            st.error(f"Cannot load system info: {system_info['error']}")
        else:
            st.text(f"🧠 Embedding: {system_info.get('embedding_model', 'Unknown')}")
            st.text(f"🦙 LLM: {system_info.get('llm_model', 'Unknown')}")

            collection = system_info.get('collection_info', {})
            st.text(f"📚 Documents: {collection.get('points', 0)}")

            gpu_info = system_info.get('gpu_info', {})
            if gpu_info.get('available'):
                st.text(f"🎮 GPU: {gpu_info.get('name', 'Unknown')}")
            else:
                st.text("💻 CPU only")


def display_examples():
    """Display example queries in the sidebar."""
    with st.sidebar.expander("💡 Example Questions", expanded=True):
        examples = get_example_queries()

        if examples:
            for category, questions in examples.items():
                st.subheader(category.replace('_', ' ').title())
                for i, question in enumerate(questions[:3]):  # Show first 3 in each category
                    # Use unique keys that won't conflict
                    button_key = f"example_{category}_{i}_{len(question)}"
                    if st.button(f"📝 {question[:50]}...", key=button_key):
                        # Set the question in session state and trigger processing
                        st.session_state.current_question = question
                        st.session_state.query_mode = "research"
                        st.session_state.top_k = 7
                        st.session_state.include_sources = True
                        st.session_state.is_processing = True
                        # Clear any existing results to avoid confusion
                        if 'last_response' in st.session_state:
                            del st.session_state.last_response
                        if 'last_sources' in st.session_state:
                            del st.session_state.last_sources
                        if 'last_question' in st.session_state:
                            del st.session_state.last_question
                        st.rerun()


def display_research_interface():
    """Main research query interface."""
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Research Assistant</strong><br>
    Ask questions about microplastic pollution in marine ecosystems. 
    I have access to recent scientific papers and can provide evidence-based answers with source citations.
    </div>
    """, unsafe_allow_html=True)

    # Query input
    col1, col2 = st.columns([4, 1])

    with col1:
        # Use selected question from examples if available, otherwise use text area input
        if 'selected_question' in st.session_state:
            default_question = st.session_state.selected_question
            del st.session_state.selected_question  # Clear it after using
        else:
            default_question = ''

        question = st.text_area(
            "🔍 Ask your research question:",
            value=default_question,
            height=100,
            placeholder="e.g., What are the main sources of microplastic pollution in marine environments?",
            key="question_input"
        )

    with col2:
        top_k = st.number_input("📚 Sources to retrieve:", min_value=1, max_value=20, value=7)
        include_sources = st.checkbox("📄 Include source details", value=True)

    # Show processing status if active
    if st.session_state.is_processing:
        st.info("🔄 Processing your request... Please wait.")

    # Query buttons
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        button_text = "⏳ Processing..." if st.session_state.is_processing else "🔍 Research Question"
        research_button = st.button(
            button_text,
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_processing,
            key="research_btn"
        )

        if research_button and not st.session_state.is_processing:
            if question.strip():
                st.session_state.is_processing = True
                st.session_state.query_mode = "research"
                st.session_state.current_question = question.strip()
                st.session_state.top_k = top_k
                st.session_state.include_sources = include_sources
                # Clear previous results
                if 'last_response' in st.session_state:
                    del st.session_state.last_response
                if 'last_sources' in st.session_state:
                    del st.session_state.last_sources
                if 'last_question' in st.session_state:
                    del st.session_state.last_question
                st.rerun()
            else:
                st.error("Please enter a question")

    with col2:
        button_text = "⏳ Processing..." if st.session_state.is_processing else "📚 Search Sources Only"
        search_button = st.button(
            button_text,
            use_container_width=True,
            disabled=st.session_state.is_processing,
            key="search_btn"
        )

        if search_button and not st.session_state.is_processing:
            if question.strip():
                st.session_state.is_processing = True
                st.session_state.query_mode = "search"
                st.session_state.current_question = question.strip()
                st.session_state.top_k = top_k
                # Clear previous results
                if 'last_response' in st.session_state:
                    del st.session_state.last_response
                if 'last_sources' in st.session_state:
                    del st.session_state.last_sources
                if 'last_question' in st.session_state:
                    del st.session_state.last_question
                st.rerun()
            else:
                st.error("Please enter a search query")

    with col3:
        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True,
            disabled=st.session_state.is_processing,
            key="clear_btn"
        )

        if clear_button:
            # Clear all session state
            keys_to_clear = ['query_mode', 'current_question', 'last_response', 'last_sources', 'last_question',
                             'is_processing']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def display_query_results():
    """Display the results of a query."""
    # Check if we have a question to process (either from button click or example click)
    if 'query_mode' not in st.session_state or not st.session_state.is_processing:
        return

    question = st.session_state.get('current_question', '')

    if st.session_state.query_mode == "research":
        # Research mode - get full answer with sources
        st.markdown(f'<div class="question-box"><strong>❓ Question:</strong> {question}</div>',
                    unsafe_allow_html=True)

        # Show loading spinner and process query
        with st.spinner("🤔 Analyzing research papers and generating response..."):
            response = query_api(
                question,
                st.session_state.get('top_k', 7),
                st.session_state.get('include_sources', True)
            )

        # Reset processing state
        st.session_state.is_processing = False

        if "error" in response:
            st.markdown(f'<div class="error-box"><strong>❌ Error:</strong> {response["error"]}</div>',
                        unsafe_allow_html=True)
        else:
            # Store response in session state for persistence
            st.session_state.last_response = response
            st.session_state.last_question = question  # Store the question too

            # Display answer
            st.markdown(f'<div class="answer-box"><strong>💡 Answer:</strong><br>{response["answer"]}</div>',
                        unsafe_allow_html=True)

            # Display performance metrics
            if "performance" in response:
                perf = response["performance"]
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f'<div class="metric-card"><strong>🔍 Retrieval</strong><br>{perf.get("retrieval_time", 0):.2f}s</div>',
                        unsafe_allow_html=True)
                with col2:
                    st.markdown(
                        f'<div class="metric-card"><strong>🧠 Generation</strong><br>{perf.get("generation_time", 0):.2f}s</div>',
                        unsafe_allow_html=True)
                with col3:
                    st.markdown(
                        f'<div class="metric-card"><strong>⏱️ Total</strong><br>{perf.get("total_time", 0):.2f}s</div>',
                        unsafe_allow_html=True)

            # Display sources
            if response.get("sources"):
                display_sources(response["sources"], "📚 Sources Used")

    elif st.session_state.query_mode == "search":
        # Search mode - just get sources
        st.markdown(f'<div class="question-box"><strong>🔍 Search Query:</strong> {question}</div>',
                    unsafe_allow_html=True)

        # Show loading spinner and process search
        with st.spinner("🔍 Searching research database..."):
            sources = search_sources(question, st.session_state.get('top_k', 10))

        # Reset processing state
        st.session_state.is_processing = False

        if sources:
            # Store sources in session state for persistence
            st.session_state.last_sources = sources
            st.session_state.last_question = question  # Store the question too
            display_sources(sources, f"📚 Found {len(sources)} Relevant Sources")
        else:
            st.warning("No relevant sources found for your query.")

    # Clear query mode after processing
    if 'query_mode' in st.session_state:
        del st.session_state.query_mode

    # Force a rerun to show the interface again
    st.rerun()


def display_persistent_results():
    """Display persistent results from previous queries."""
    # Show last response if it exists
    if 'last_response' in st.session_state:
        response = st.session_state.last_response
        question = st.session_state.get('last_question', 'Previous question')

        # Show the question that was asked
        st.markdown(f'<div class="question-box"><strong>❓ Last Question:</strong> {question}</div>',
                    unsafe_allow_html=True)

        st.markdown(f'<div class="answer-box"><strong>💡 Answer:</strong><br>{response["answer"]}</div>',
                    unsafe_allow_html=True)

        # Display performance metrics
        if "performance" in response:
            perf = response["performance"]
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f'<div class="metric-card"><strong>🔍 Retrieval</strong><br>{perf.get("retrieval_time", 0):.2f}s</div>',
                    unsafe_allow_html=True)
            with col2:
                st.markdown(
                    f'<div class="metric-card"><strong>🧠 Generation</strong><br>{perf.get("generation_time", 0):.2f}s</div>',
                    unsafe_allow_html=True)
            with col3:
                st.markdown(
                    f'<div class="metric-card"><strong>⏱️ Total</strong><br>{perf.get("total_time", 0):.2f}s</div>',
                    unsafe_allow_html=True)

        # Display sources
        if response.get("sources"):
            display_sources(response["sources"], "📚 Sources Used")

    # Show last search sources if they exist
    elif 'last_sources' in st.session_state:
        sources = st.session_state.last_sources
        question = st.session_state.get('last_question', 'Previous search')

        # Show the search query that was used
        st.markdown(f'<div class="question-box"><strong>🔍 Last Search:</strong> {question}</div>',
                    unsafe_allow_html=True)

        display_sources(sources, f"📚 Found {len(sources)} Relevant Sources")


def display_sources(sources: List[Dict[str, Any]], title: str):
    """Display a list of sources in an organized way."""
    st.subheader(title)

    for i, source in enumerate(sources, 1):
        with st.expander(f"📄 {i}. {source['title'][:80]}{'...' if len(source['title']) > 80 else ''}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Title:** {source['title']}")

                if source.get('authors'):
                    authors_str = ", ".join(source['authors'][:3])
                    if len(source['authors']) > 3:
                        authors_str += " et al."
                    st.markdown(f"**Authors:** {authors_str}")

                if source.get('journal'):
                    st.markdown(f"**Journal:** {source['journal']}")

                if source.get('url'):
                    st.markdown(f"**URL:** [Link to paper]({source['url']})")

                st.markdown(f"**Content Type:** {source.get('content_type', 'Unknown')}")
                st.markdown(f"**Relevance Score:** {source.get('score', 0):.3f}")

            with col2:
                st.metric("Chunk #", source.get('chunk_index', 0))
                st.metric("Relevance", f"{source.get('score', 0):.1%}")

            # Show text preview
            if source.get('text_preview'):
                st.markdown("**Content Preview:**")
                st.text(source['text_preview'])


def display_statistics():
    """Display usage statistics and system status."""
    st.subheader("📊 System Statistics")

    system_info = get_system_info()
    if "error" not in system_info:
        col1, col2, col3, col4 = st.columns(4)

        collection = system_info.get('collection_info', {})

        with col1:
            st.metric("📄 Documents", collection.get('points', 0))

        with col2:
            st.metric("🧠 Vector Size", collection.get('vectors', 0))

        with col3:
            gpu_info = system_info.get('gpu_info', {})
            gpu_status = "🎮 Available" if gpu_info.get('available') else "💻 CPU Only"
            st.metric("Hardware", gpu_status)

        with col4:
            config = system_info.get('configuration', {})
            st.metric("🔍 Max Results", config.get('top_k_results', 'Unknown'))


def main():
    """Main Streamlit application."""
    # Initialize session state
    if 'query_mode' not in st.session_state:
        st.session_state.query_mode = None
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    # Sidebar
    with st.sidebar:
        st.markdown("## 🌊 Microplastics Research")
        st.markdown("*AI-Powered Research Assistant*")

        st.markdown("---")

        display_health_status()
        display_system_info()
        display_examples()

    # Main content area - SINGLE INTERFACE
    st.markdown('<div class="main-header"><h1>🌊 Microplastics Research Assistant</h1></div>',
                unsafe_allow_html=True)

    # Create tabs but ensure only one interface
    tab1, tab2, tab3 = st.tabs(["🔬 Research", "📊 Statistics", "ℹ️ About"])

    with tab1:
        # Check if we should process an example question first
        if st.session_state.is_processing and 'current_question' in st.session_state:
            display_query_results()

        # Always show the interface (but process results first if needed)
        if not st.session_state.is_processing:
            display_research_interface()

        # Show persistent results from previous queries
        if not st.session_state.is_processing:
            display_persistent_results()

    with tab2:
        display_statistics()

        # Additional system metrics
        st.subheader("🔧 System Health")
        health = check_api_health()

        if health.get("status") == "healthy":
            st.success("✅ All systems operational")
        elif health.get("status") == "degraded":
            st.warning("⚠️ Some issues detected")
        else:
            st.error("❌ System unavailable")

        # Component status
        if "components" in health:
            st.subheader("🔍 Component Status")
            for component, status in health["components"].items():
                if status.startswith("ready"):
                    st.success(f"✅ {component.replace('_', ' ').title()}: {status}")
                elif "error" in status:
                    st.error(f"❌ {component.replace('_', ' ').title()}: {status}")
                else:
                    st.warning(f"⚠️ {component.replace('_', ' ').title()}: {status}")

    with tab3:
        st.markdown("""
        ## 🌊 About Microplastics Research Assistant

        This AI-powered research assistant helps you explore scientific literature about microplastic pollution in marine ecosystems.

        ### 🔬 Features
        - **Intelligent Search**: Uses advanced vector similarity to find relevant research papers
        - **AI-Generated Answers**: Provides comprehensive answers grounded in scientific literature
        - **Source Citations**: Always includes references to the original research papers
        - **Real-time Processing**: Fast responses powered by GPU acceleration (when available)

        ### 📚 Knowledge Base
        The system has access to recent scientific papers (2020-2024) covering:
        - Sources and pathways of microplastic pollution
        - Environmental distribution and fate of microplastics
        - Biological impacts on marine organisms
        - Detection and analysis methods
        - Mitigation and prevention strategies

        ### 🔧 Technology Stack
        - **Backend**: FastAPI with async processing
        - **Frontend**: Streamlit for interactive UI
        - **Vector Database**: Qdrant for similarity search
        - **Embeddings**: Sentence Transformers for text encoding
        - **LLM**: Llama 3 via Ollama for response generation
        - **GPU Acceleration**: CUDA support for faster processing

        ### 💡 How to Use
        1. **Ask Questions**: Type your research question in natural language
        2. **Review Sources**: Check the cited papers for detailed information
        3. **Refine Queries**: Use more specific terms for targeted results
        4. **Explore Examples**: Try the suggested questions in the sidebar

        ### ⚠️ Important Notes
        - Answers are based on the available literature in the database
        - Always verify information by checking the original sources
        - The system provides scientific information, not policy recommendations
        - For the most current research, supplement with recent literature searches

        ### 🤝 Contributing
        This is an open-source research tool. Contributions and feedback are welcome!

        ---
        *Built for researchers, by researchers* 🧬
        """)

        # System requirements and setup info
        with st.expander("🔧 Technical Details", expanded=False):
            st.markdown("""
            **System Requirements:**
            - Python 3.8+
            - CUDA-compatible GPU (recommended)
            - 8GB+ RAM
            - Internet connection for initial setup

            **Setup Process:**
            1. Install dependencies: `pip install -r requirements.txt`
            2. Configure environment: Copy `.env.template` to `.env`
            3. Start Ollama: `ollama serve`
            4. Install model: `ollama pull llama3:8b`
            5. Collect data: `python scripts/collect_data.py`
            6. Process documents: `python scripts/process_documents.py`
            7. Index to Qdrant: `python scripts/index_to_qdrant.py`
            8. Start API: `python app/api.py`
            9. Start UI: `streamlit run app/streamlit_app.py`

            **Performance Tips:**
            - Use GPU acceleration for faster embedding generation
            - Increase batch sizes if you have more VRAM
            - Monitor system resources during processing
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        "🌊 **Microplastics Research Assistant** | "
        "Powered by AI for Marine Science Research | "
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    )


if __name__ == "__main__":
    main()