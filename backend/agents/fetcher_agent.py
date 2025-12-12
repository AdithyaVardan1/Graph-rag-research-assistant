"""
Fetcher Agent - Fetches research papers from arXiv with rate limit handling.
"""
import time
import logging
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# arXiv recommends no more than 1 request every 3 seconds
ARXIV_RATE_LIMIT_DELAY = 3
MAX_RETRIES = 3


class FetcherAgent:
    """
    Fetch research papers from arXiv based on a query.
    Includes retry logic for rate limiting.
    """
    
    def __init__(self, base_url: str = "http://export.arxiv.org/api/query"):
        self.base_url = base_url
        self.last_request_time = 0

    def _wait_for_rate_limit(self):
        """Ensure we wait at least 3 seconds between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < ARXIV_RATE_LIMIT_DELAY:
            wait_time = ARXIV_RATE_LIMIT_DELAY - elapsed
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s before arXiv request")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def fetch_papers(self, query: str, max_results: int = 10, years: Optional[List[int]] = None) -> List[Dict]:
        """
        Fetch papers from arXiv API for the given query.
        Includes retry logic for rate limit errors (429, 503).
        """
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance"
        }

        headers = {
            'User-Agent': 'GraphRAGResearchAssistant/1.0 (https://github.com/research-assistant; mailto:your@email.com)'
        }

        # Retry loop for rate limiting
        for attempt in range(MAX_RETRIES):
            self._wait_for_rate_limit()
            
            try:
                response = requests.get(self.base_url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    papers = self._parse_response(response.text)
                    if years:
                        years_str = set(str(y) for y in years)
                        papers = [p for p in papers if p['published'][:4] in years_str]
                    return papers
                    
                elif response.status_code in [429, 503]:
                    # Rate limited or service unavailable - wait and retry
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    logger.warning(f"arXiv rate limit ({response.status_code}). Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    raise Exception(f"arXiv API error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"arXiv request timeout. Retry {attempt + 1}/{MAX_RETRIES}")
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching papers: {e}")
                raise

        # All retries exhausted
        raise Exception("arXiv API rate limit exceeded. Please wait a minute and try again.")

    def _parse_response(self, xml_text: str) -> List[Dict]:
        """
        Parse XML response from arXiv and return structured data.
        """
        root = ET.fromstring(xml_text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        
        for entry in root.findall('atom:entry', ns):
            paper = {
                'id': entry.find('atom:id', ns).text.strip(),
                'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                'summary': entry.find('atom:summary', ns).text.strip().replace('\n', ' '),
                'published': entry.find('atom:published', ns).text,
                'link': entry.find('atom:id', ns).text,
                'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            }
            papers.append(paper)
            
        return papers