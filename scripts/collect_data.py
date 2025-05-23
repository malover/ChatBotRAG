import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Microplastics research paper URLs (open access - updated for better success)
PAPER_URLS = [
    # PMC (PubMed Central) - usually more accessible
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7927104/",  # Microplastics in the Marine Environment
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8789065/",  # Marine Microplastics
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6471892/",  # Microplastics in Marine Food Webs

    # Frontiers (Open Access)
    "https://www.frontiersin.org/articles/10.3389/fmars.2020.00308/full",

    # PLoS ONE (Open Access)
    "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236509",  # Microplastics in Mediterranean Sea

    # MDPI alternatives (try different approach)
    "https://www.mdpi.com/2071-1050/15/17/13252/htm",  # Try htm version
    "https://www.mdpi.com/2073-4441/14/19/3088/htm",  # Try htm version
]


class MicroplasticsDataCollector:
    """Collects and processes microplastics research data."""

    def __init__(self):
        self.session = requests.Session()
        # Use different user agents and headers to avoid blocking
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]

        self.session.headers.update({
            'User-Agent': user_agents[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def scrape_paper(self, url: str) -> Optional[Dict]:
        """Scrape a single research paper with retry logic."""
        logger.info(f"Scraping: {url}")

        # Try different approaches for different sites
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Add some randomization to avoid detection
                if attempt > 0:
                    import random
                    time.sleep(random.uniform(3, 7))

                    # Try different user agent
                    user_agents = [
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
                    ]
                    import random
                    self.session.headers['User-Agent'] = random.choice(user_agents)

                response = self.session.get(url, timeout=30)

                # Handle different response codes
                if response.status_code == 403:
                    logger.warning(f"403 Forbidden for {url}, attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error(f"Failed to access {url} - site may be blocking automated requests")
                        return None

                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                domain = urlparse(url).netloc.lower()

                # Extract paper information
                paper_data = {
                    'url': url,
                    'domain': domain,
                    'title': self._extract_title(soup, domain),
                    'abstract': self._extract_abstract(soup, domain),
                    'content': self._extract_content(soup, domain),
                    'authors': self._extract_authors(soup, domain),
                    'journal': self._extract_journal(soup, domain),
                    'date': self._extract_date(soup, domain),
                    'keywords': self._extract_keywords(soup, domain)
                }

                # Validate that we got meaningful content
                if len(paper_data['content']) < 500:
                    logger.warning(
                        f"Content too short for {url} ({len(paper_data['content'])} chars), might be blocked or paywall")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return None

                logger.info(f"Successfully scraped {url} ({len(paper_data['content'])} chars)")
                return paper_data

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {url}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error(f"Failed to scrape {url} after {max_retries} attempts")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error scraping {url}: {e}")
                return None

        return None

    def _extract_title(self, soup: BeautifulSoup, domain: str) -> str:
        """Extract paper title."""
        selectors = [
            'h1.title', 'h1.article-title', 'h1[data-test="article-title"]',
            '.c-article-title', '.JournalFullText h1', 'h1', 'title'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if len(title) > 10 and not title.lower().startswith('error'):
                    return title

        return "Title not found"

    def _extract_abstract(self, soup: BeautifulSoup, domain: str) -> str:
        """Extract paper abstract."""
        selectors = [
            '.abstract', '#abstract', '.article-abstract',
            '.abstractInFull', '.JournalAbstract', '[data-test="abstract"]',
            '.c-article-section__content'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                abstract = element.get_text(separator=' ', strip=True)
                if len(abstract) > 100:
                    return abstract

        return ""

    def _extract_content(self, soup: BeautifulSoup, domain: str) -> str:
        """Extract main paper content."""
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
            tag.decompose()

        # Remove common non-content elements
        for class_name in ['navigation', 'sidebar', 'comments', 'references', 'citation']:
            for elem in soup.find_all(class_=class_name):
                elem.decompose()

        # Domain-specific content selectors
        content_selectors = {
            'ncbi.nlm.nih.gov': ['.article', '.formatted', 'main', '#maincontent', '.content'],
            'frontiersin.org': ['.JournalFullText', '.article-container', 'main', 'article'],
            'plos.org': ['.article-content', '#artText', '.article-text'],
            'nature.com': ['#content', '.article-body', 'main'],
            'mdpi.com': ['.article-content', '.html', 'article', '#article_body'],
            'default': ['article', 'main', '.content', '.article-body', '.paper-content']
        }

        selectors = content_selectors.get(domain, content_selectors['default'])

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # For PMC articles, also try to get specific sections
                if 'ncbi.nlm.nih.gov' in domain:
                    sections = element.find_all(['div', 'section'], class_=['sec', 'section'])
                    if sections:
                        section_texts = []
                        for section in sections:
                            section_text = section.get_text(separator='\n', strip=True)
                            if len(section_text) > 100:  # Only include substantial sections
                                section_texts.append(section_text)
                        if section_texts:
                            content = '\n\n'.join(section_texts)
                        else:
                            content = element.get_text(separator='\n', strip=True)
                    else:
                        content = element.get_text(separator='\n', strip=True)
                else:
                    content = element.get_text(separator='\n', strip=True)

                # Clean up the content
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                content = '\n'.join(lines)

                # Remove very short lines that are likely navigation/metadata
                meaningful_lines = []
                for line in lines:
                    if len(line) > 20 or any(keyword in line.lower() for keyword in
                                             ['microplastic', 'plastic', 'marine', 'ocean', 'pollution']):
                        meaningful_lines.append(line)

                if meaningful_lines:
                    content = '\n'.join(meaningful_lines)

                if len(content) > 1000:
                    return content

        return ""

    def _extract_authors(self, soup: BeautifulSoup, domain: str) -> List[str]:
        """Extract author names."""
        selectors = [
            '.author-name', '.authors .author', '.c-article-author',
            '[data-test="author-name"]', '.contrib-group .contrib'
        ]

        authors = []
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                authors = [elem.get_text(strip=True) for elem in elements[:10]]
                break

        return [author for author in authors if author and len(author) > 2]

    def _extract_journal(self, soup: BeautifulSoup, domain: str) -> str:
        """Extract journal name."""
        selectors = [
            'meta[name="citation_journal_title"]',
            'meta[property="og:site_name"]',
            '.journal-title', '.journal-name'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    return element.get('content', '')
                else:
                    return element.get_text(strip=True)

        return ""

    def _extract_date(self, soup: BeautifulSoup, domain: str) -> str:
        """Extract publication date."""
        selectors = [
            'meta[name="citation_publication_date"]',
            'meta[property="article:published_time"]',
            'time[datetime]', '.publication-date'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    return element.get('content', '')
                elif element.name == 'time':
                    return element.get('datetime', element.get_text(strip=True))
                else:
                    return element.get_text(strip=True)

        return ""

    def _extract_keywords(self, soup: BeautifulSoup, domain: str) -> List[str]:
        """Extract keywords."""
        selectors = [
            'meta[name="citation_keywords"]',
            'meta[name="keywords"]',
            '.keywords'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    keywords_str = element.get('content', '')
                else:
                    keywords_str = element.get_text(strip=True)

                if keywords_str:
                    keywords = [kw.strip() for kw in keywords_str.replace(';', ',').split(',')]
                    return [kw for kw in keywords if kw and len(kw) > 2]

        return []

    def save_paper(self, paper_data: Dict, filename: str):
        """Save paper data to JSON file."""
        filepath = RAW_DIR / f"{filename}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(paper_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved: {filepath}")

    def collect_all_papers(self) -> List[Dict]:
        """Collect all papers from the URL list."""
        collected_papers = []

        logger.info(f"Starting to collect {len(PAPER_URLS)} papers...")

        for i, url in enumerate(tqdm(PAPER_URLS, desc="Collecting papers")):
            paper_data = self.scrape_paper(url)

            if paper_data:
                # Create filename from domain and index
                domain = urlparse(url).netloc.replace('.', '_')
                filename = f"paper_{i + 1:02d}_{domain}"

                self.save_paper(paper_data, filename)
                collected_papers.append(paper_data)

            # Be respectful to servers
            time.sleep(2)

        logger.info(f"Successfully collected {len(collected_papers)} papers")
        return collected_papers


class DataProcessor:
    """Process collected papers for analysis."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_papers(self) -> List[Dict]:
        """Load all collected papers."""
        papers = []
        json_files = list(RAW_DIR.glob("*.json"))

        logger.info(f"Loading {len(json_files)} papers...")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    papers.append(paper)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        return papers

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                if last_period != -1:
                    end = last_period + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def process_papers(self) -> List[Dict]:
        """Process all papers into chunks."""
        papers = self.load_papers()
        processed_chunks = []

        logger.info(f"Processing {len(papers)} papers into chunks...")

        for paper in tqdm(papers, desc="Processing papers"):
            # Combine title, abstract, and content
            full_text = ""

            if paper.get('title'):
                full_text += f"Title: {paper['title']}\n\n"

            if paper.get('abstract'):
                full_text += f"Abstract: {paper['abstract']}\n\n"

            if paper.get('content'):
                full_text += f"Content: {paper['content']}"

            # Create chunks
            chunks = self.create_chunks(full_text)

            # Create chunk documents
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': f"{paper['url']}#chunk_{i}",
                    'text': chunk,
                    'metadata': {
                        'title': paper.get('title', ''),
                        'authors': paper.get('authors', []),
                        'journal': paper.get('journal', ''),
                        'date': paper.get('date', ''),
                        'url': paper.get('url', ''),
                        'keywords': paper.get('keywords', []),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'source_type': 'scientific_paper'
                    }
                }
                processed_chunks.append(chunk_data)

        # Save processed chunks
        output_file = PROCESSED_DIR / "processed_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Created {len(processed_chunks)} chunks from {len(papers)} papers")
        logger.info(f"Saved processed data to: {output_file}")

        return processed_chunks

    def get_statistics(self, chunks: List[Dict]) -> Dict:
        """Get statistics about processed data."""
        if not chunks:
            return {}

        total_chunks = len(chunks)
        unique_papers = len(set(chunk['metadata']['url'] for chunk in chunks))
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_chunk_size = total_chars / total_chunks

        # Get journals and dates
        journals = set()
        dates = set()
        authors = set()

        for chunk in chunks:
            metadata = chunk['metadata']
            if metadata.get('journal'):
                journals.add(metadata['journal'])
            if metadata.get('date'):
                dates.add(metadata['date'])
            for author in metadata.get('authors', []):
                authors.add(author)

        stats = {
            'total_chunks': total_chunks,
            'unique_papers': unique_papers,
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 2),
            'unique_journals': len(journals),
            'unique_authors': len(authors),
            'date_range': sorted(list(dates)),
            'journals': sorted(list(journals))
        }

        return stats


def main():
    """Main function to collect and process data."""
    print("🌊 Microplastics Research Data Collector")
    print("=" * 50)

    # Step 1: Collect papers
    collector = MicroplasticsDataCollector()
    papers = collector.collect_all_papers()

    if not papers:
        print("❌ No papers were successfully collected. Check URLs and network connection.")
        return

    print(f"✅ Successfully collected {len(papers)} papers")

    # Step 2: Process papers
    print("\n" + "=" * 50)
    print("Processing papers into chunks...")

    processor = DataProcessor()
    chunks = processor.process_papers()

    if not chunks:
        print("❌ No chunks were created. Check the processing step.")
        return

    print(f"✅ Successfully created {len(chunks)} chunks")

    # Step 3: Show statistics
    print("\n" + "=" * 50)
    print("📊 Data Statistics:")

    stats = processor.get_statistics(chunks)
    for key, value in stats.items():
        if isinstance(value, list) and len(value) > 5:
            print(f"  {key}: {value[:5]}... ({len(value)} total)")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("✅ Data collection and processing complete!")
    print(f"📁 Raw papers saved in: {RAW_DIR}")
    print(f"📁 Processed chunks saved in: {PROCESSED_DIR}")
    print("\nNext steps:")
    print("1. Review the collected data in the data/ directory")
    print("2. Check data quality and content")
    print("3. Proceed with vector database setup")


if __name__ == "__main__":
    main()