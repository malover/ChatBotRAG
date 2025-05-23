import sys
import time
from typing import Optional
from pathlib import Path

# Add root directory to path for imports
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.rag_system import MicroplasticsRAG, RAGResponse


class InteractiveChat:
    """Simple command-line chat interface."""

    def __init__(self):
        self.rag_system: Optional[MicroplasticsRAG] = None
        self.conversation_history = []

    def initialize(self) -> bool:
        """Initialize the RAG system."""
        print("🔄 Initializing RAG system...")

        try:
            self.rag_system = MicroplasticsRAG()

            # Get and display system info
            system_info = self.rag_system.get_system_info()
            print(f"✅ RAG system initialized successfully!")
            print(f"📊 System Info:")
            print(f"   📚 Knowledge Base: {system_info.get('collection', {}).get('points', 0)} documents")
            print(f"   🧠 Embedding Model: {system_info.get('embedding_model', 'Unknown')}")
            print(f"   🦙 LLM: {system_info.get('llm_model', 'Unknown')}")
            print(f"   🎮 GPU: {system_info.get('gpu_name', 'CPU')}")

            return True

        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            print(f"\n🔧 Make sure:")
            print(f"   1. Ollama is running: ollama serve")
            print(f"   2. Model is installed: ollama pull llama3:8b")
            print(f"   3. Qdrant has data: python index_to_qdrant.py")
            return False

    def print_banner(self):
        """Print the chat banner."""
        print("\n" + "🌊" * 25)
        print("🌊 MICROPLASTICS RESEARCH ASSISTANT 🌊")
        print("🌊" * 25)
        print(f"💬 Ask me about microplastic pollution in marine ecosystems!")
        print(f"📚 I have access to recent scientific papers (2020-2024)")
        print(f"\n💡 Example questions:")
        print(f"   • What are microplastics and where do they come from?")
        print(f"   • How do microplastics affect marine life?")
        print(f"   • What methods detect microplastics in seawater?")
        print(f"   • How can we reduce microplastic pollution?")
        print(f"\n⌨️  Commands:")
        print(f"   • 'help' - Show this help")
        print(f"   • 'stats' - Show system statistics")
        print(f"   • 'history' - Show conversation history")
        print(f"   • 'clear' - Clear conversation history")
        print(f"   • 'quit' or 'exit' - Exit the chat")
        print("=" * 60)

    def format_response(self, response: RAGResponse) -> str:
        """Format the RAG response for display."""
        output = []

        # Main answer
        output.append(f"💡 {response.answer}")

        # Sources
        if response.sources:
            output.append(f"\n📚 Sources:")
            for i, source in enumerate(response.sources, 1):
                relevance = "🎯" if source.score > 0.8 else "📄"
                output.append(f"   {relevance} {i}. {source.title}")

                if source.authors:
                    authors_str = ", ".join(source.authors[:2])
                    if len(source.authors) > 2:
                        authors_str += " et al."
                    output.append(f"      Authors: {authors_str}")

                if source.journal:
                    output.append(f"      Journal: {source.journal}")

                output.append(f"      Relevance: {source.score:.1%}")
                output.append("")  # Empty line

        # Performance metrics
        output.append(f"⏱️  Response time: {response.total_time:.2f}s")

        return "\n".join(output)

    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        command = command.lower().strip()

        if command == 'help':
            self.print_banner()
            return True

        elif command == 'stats':
            if self.rag_system:
                system_info = self.rag_system.get_system_info()
                print(f"\n📊 System Statistics:")
                print(f"   Knowledge Base: {system_info.get('collection', {}).get('points', 0)} documents")
                print(f"   Vectors: {system_info.get('collection', {}).get('vectors', 0)}")
                print(f"   Embedding Model: {system_info.get('embedding_model', 'Unknown')}")
                print(f"   LLM Model: {system_info.get('llm_model', 'Unknown')}")
                print(f"   GPU Available: {system_info.get('gpu_available', False)}")
                print(f"   Top-K Results: {system_info.get('configuration', {}).get('top_k_results', 'Unknown')}")
                print(
                    f"   Similarity Threshold: {system_info.get('configuration', {}).get('similarity_threshold', 'Unknown')}")
            return True

        elif command == 'history':
            if self.conversation_history:
                print(f"\n📜 Conversation History ({len(self.conversation_history)} exchanges):")
                for i, (question, answer_preview) in enumerate(self.conversation_history, 1):
                    print(f"   {i}. Q: {question}")
                    print(f"      A: {answer_preview[:100]}...")
                    print()
            else:
                print(f"\n📜 No conversation history yet.")
            return True

        elif command == 'clear':
            self.conversation_history.clear()
            print(f"\n🗑️  Conversation history cleared.")
            return True

        elif command in ['quit', 'exit', 'bye']:
            print(f"\n👋 Thanks for using the Microplastics Research Assistant!")
            print(f"🌊 Keep our oceans clean! 🌊")
            return True

        return False

    def chat_loop(self):
        """Main chat interaction loop."""
        self.print_banner()

        while True:
            try:
                # Get user input
                user_input = input(f"\n🔬 Ask about microplastics: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if self.handle_command(user_input):
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    continue

                # Process as RAG query
                print(f"\n🤔 Thinking...")
                start_time = time.time()

                response = self.rag_system.query(user_input)

                # Display response
                print(f"\n" + "=" * 60)
                formatted_response = self.format_response(response)
                print(formatted_response)
                print(f"=" * 60)

                # Save to history
                answer_preview = response.answer[:100].replace('\n', ' ')
                self.conversation_history.append((user_input, answer_preview))

            except KeyboardInterrupt:
                print(f"\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing your question: {e}")
                print(f"💡 Try rephrasing your question or type 'help' for assistance.")


def main():
    """Main function to run the interactive chat."""
    chat = InteractiveChat()

    # Initialize the system
    if not chat.initialize():
        print(f"\n🔧 Setup required before using the chat interface:")
        print(f"1. Start Ollama: ollama serve")
        print(f"2. Install model: ollama pull llama3:8b")
        print(f"3. Index documents: python index_to_qdrant.py")
        print(f"4. Test system: python -m src.rag_system")
        sys.exit(1)

    # Start the chat
    try:
        chat.chat_loop()
    except Exception as e:
        print(f"\n❌ Chat interface error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()