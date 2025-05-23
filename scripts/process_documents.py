import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Load configuration from .env
from config_rtx4080 import (
    CHUNK_SIZE, CHUNK_OVERLAP, validate_config
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


class DocumentPreprocessor:
    """Preprocesses documents for vector indexing."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized preprocessor: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def load_collected_papers(self) -> List[Dict]:
        """Load all collected papers from the raw directory."""
        papers = []
        json_files = list(RAW_DIR.glob("*.json"))

        if not json_files:
            logger.error(f"No JSON files found in {RAW_DIR}")
            logger.info("Run collect_data.py first to gather research papers")
            return papers

        logger.info(f"Loading {len(json_files)} papers from {RAW_DIR}")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    papers.append(paper)
                    logger.debug(f"Loaded {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        logger.info(f"Successfully loaded {len(papers)} papers")
        return papers

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks with smart boundary detection."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at good boundaries
            if end < len(text):
                # Look for sentence ending within the last 150 characters
                boundary_found = False

                # Try different sentence endings in order of preference
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_punct = text.rfind(punct, max(start, end - 150), end)
                    if last_punct != -1:
                        end = last_punct + len(punct)
                        boundary_found = True
                        break

                # If no sentence boundary, try paragraph break
                if not boundary_found:
                    last_para = text.rfind('\n\n', max(start, end - 100), end)
                    if last_para != -1:
                        end = last_para + 2
                        boundary_found = True

                # If still no good boundary, try single newline
                if not boundary_found:
                    last_newline = text.rfind('\n', max(start, end - 50), end)
                    if last_newline != -1:
                        end = last_newline + 1

            chunk = text[start:end].strip()

            # Only include substantial chunks
            if chunk and len(chunk) > 50:
                chunks.append(chunk)
                logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} chars")

            # Move start position with overlap
            if end >= len(text):
                break
            start = end - self.chunk_overlap

        return chunks

    def preprocess_paper(self, paper: Dict, paper_idx: int) -> List[Dict]:
        """Preprocess a single paper into chunks."""
        # Combine different parts of the paper
        sections = []

        # Add title with clear labeling
        if paper.get('title'):
            sections.append(f"Title: {paper['title']}")

        # Add abstract with clear labeling
        if paper.get('abstract'):
            sections.append(f"Abstract: {paper['abstract']}")

        # Add main content
        if paper.get('content'):
            # Clean the content first
            content = paper['content']
            # Remove excessive whitespace
            content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
            sections.append(f"Full Text: {content}")

        # Combine all sections
        full_text = "\n\n".join(sections)

        if len(full_text) < 100:
            logger.warning(f"Paper {paper_idx} has very little content ({len(full_text)} chars)")
            return []

        # Create chunks
        text_chunks = self.create_chunks(full_text)

        # Create chunk documents with rich metadata
        chunk_docs = []
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_doc = {
                'id': f"paper_{paper_idx}_chunk_{chunk_idx}",
                'text': chunk_text,
                'metadata': {
                    'paper_id': paper_idx,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks),
                    'title': paper.get('title', 'Unknown Title'),
                    'authors': paper.get('authors', []),
                    'journal': paper.get('journal', ''),
                    'publication_date': paper.get('date', ''),
                    'url': paper.get('url', ''),
                    'keywords': paper.get('keywords', []),
                    'domain': paper.get('domain', ''),
                    'chunk_size': len(chunk_text),
                    'has_title': 'Title:' in chunk_text[:100],
                    'has_abstract': 'Abstract:' in chunk_text[:200],
                    'content_type': self._classify_chunk_content(chunk_text)
                }
            }
            chunk_docs.append(chunk_doc)

        logger.debug(f"Paper {paper_idx} -> {len(chunk_docs)} chunks")
        return chunk_docs

    def _classify_chunk_content(self, text: str) -> str:
        """Classify the type of content in a chunk."""
        text_lower = text.lower()

        if text.startswith('Title:'):
            return 'title_section'
        elif text.startswith('Abstract:'):
            return 'abstract_section'
        elif text.startswith('Full Text:'):
            return 'main_content'
        elif any(keyword in text_lower for keyword in ['introduction', 'background']):
            return 'introduction'
        elif any(keyword in text_lower for keyword in ['method', 'experimental', 'materials']):
            return 'methodology'
        elif any(keyword in text_lower for keyword in ['result', 'findings', 'data']):
            return 'results'
        elif any(keyword in text_lower for keyword in ['discussion', 'analysis']):
            return 'discussion'
        elif any(keyword in text_lower for keyword in ['conclusion', 'summary']):
            return 'conclusion'
        elif any(keyword in text_lower for keyword in ['reference', 'bibliography']):
            return 'references'
        else:
            return 'general_content'

    def preprocess_all_papers(self) -> List[Dict]:
        """Preprocess all papers into chunks with metadata."""
        papers = self.load_collected_papers()

        if not papers:
            return []

        all_chunks = []
        failed_papers = 0

        logger.info("Preprocessing papers into chunks...")

        for paper_idx, paper in enumerate(tqdm(papers, desc="Processing papers")):
            try:
                chunk_docs = self.preprocess_paper(paper, paper_idx)
                if chunk_docs:
                    all_chunks.extend(chunk_docs)
                else:
                    failed_papers += 1
                    logger.warning(f"Paper {paper_idx} produced no chunks")
            except Exception as e:
                failed_papers += 1
                logger.error(f"Error processing paper {paper_idx}: {e}")

        logger.info(f"Preprocessing complete:")
        logger.info(f"  - {len(papers)} papers processed")
        logger.info(f"  - {len(all_chunks)} chunks created")
        logger.info(f"  - {failed_papers} papers failed")

        # Save processed chunks
        if all_chunks:
            self._save_processed_chunks(all_chunks)

        return all_chunks

    def _save_processed_chunks(self, chunks: List[Dict]):
        """Save processed chunks to file."""
        PROCESSED_DIR.mkdir(exist_ok=True)
        output_file = PROCESSED_DIR / "processed_chunks.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(chunks)} processed chunks to {output_file}")

        # Also save processing statistics
        stats = self._calculate_statistics(chunks)
        stats_file = PROCESSED_DIR / "processing_stats.json"

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processing statistics to {stats_file}")

    def _calculate_statistics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate processing statistics."""
        if not chunks:
            return {}

        total_chunks = len(chunks)
        unique_papers = len(set(chunk['metadata']['paper_id'] for chunk in chunks))
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_chunk_size = total_chars / total_chunks

        # Content type distribution
        content_types = {}
        for chunk in chunks:
            content_type = chunk['metadata']['content_type']
            content_types[content_type] = content_types.get(content_type, 0) + 1

        # Chunk size distribution
        chunk_sizes = [len(chunk['text']) for chunk in chunks]

        # Journal distribution
        journals = {}
        for chunk in chunks:
            journal = chunk['metadata']['journal']
            if journal:
                journals[journal] = journals.get(journal, 0) + 1

        stats = {
            'processing_config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            },
            'totals': {
                'unique_papers': unique_papers,
                'total_chunks': total_chunks,
                'total_characters': total_chars,
                'average_chunk_size': round(avg_chunk_size, 2)
            },
            'chunk_size_distribution': {
                'min': min(chunk_sizes),
                'max': max(chunk_sizes),
                'median': sorted(chunk_sizes)[len(chunk_sizes) // 2]
            },
            'content_type_distribution': content_types,
            'journal_distribution': journals,
            'top_keywords': self._extract_top_keywords(chunks)
        }

        return stats

    def _extract_top_keywords(self, chunks: List[Dict], top_n: int = 20) -> List[tuple]:
        """Extract top keywords from all chunks."""
        keyword_counts = {}

        for chunk in chunks:
            for keyword in chunk['metadata']['keywords']:
                if keyword and len(keyword) > 2:
                    keyword_lower = keyword.lower()
                    keyword_counts[keyword_lower] = keyword_counts.get(keyword_lower, 0) + 1

        # Sort by frequency and return top N
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]


def main():
    """Main preprocessing function."""
    print("📄 Microplastics Research - Document Preprocessing")
    print("=" * 60)

    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        print("❌ Configuration Errors:")
        for error in config_errors:
            print(f"   - {error}")
        print("\n💡 Please fix your .env file first")
        return

    print(f"⚙️  Configuration:")
    print(f"   Chunk size: {CHUNK_SIZE} characters")
    print(f"   Chunk overlap: {CHUNK_OVERLAP} characters")

    try:
        # Initialize preprocessor and run
        preprocessor = DocumentPreprocessor()
        chunks = preprocessor.preprocess_all_papers()

        if not chunks:
            print("❌ No chunks were created.")
            print("💡 Check that you have data in data/raw/ directory")
            print("   Run collect_data.py if you haven't collected data yet")
            return

        print(f"\n✅ Preprocessing completed successfully!")
        print(f"📊 Results:")
        print(f"   📄 Papers: {len(set(chunk['metadata']['paper_id'] for chunk in chunks))}")
        print(f"   🧩 Chunks: {len(chunks)}")
        print(f"   📝 Total characters: {sum(len(chunk['text']) for chunk in chunks):,}")
        print(f"   📏 Avg chunk size: {sum(len(chunk['text']) for chunk in chunks) / len(chunks):.0f} chars")

        # Show content type distribution
        content_types = {}
        for chunk in chunks:
            content_type = chunk['metadata']['content_type']
            content_types[content_type] = content_types.get(content_type, 0) + 1

        print(f"\n📋 Content Distribution:")
        for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {content_type}: {count} chunks")

        print(f"\n💾 Files saved:")
        print(f"   📁 data/processed/processed_chunks.json")
        print(f"   📊 data/processed/processing_stats.json")

        print(f"\n🎯 Next step: Run index_to_qdrant.py to upload to vector database")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()