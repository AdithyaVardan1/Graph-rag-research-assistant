import requests
import xml.etree.ElementTree as ET
from typing import List, Dict

class FetcherAgent:
    """
    Fetch research papers from arXiv based on a query.
    """
    def __init__(self, base_url: str = "http://export.arxiv.org/api/query"):
        self.base_url = base_url

    def fetch_papers(self, query: str, max_results: int = 10, years: List[int] = None) -> List[Dict]:
        """
        Fetch papers from arXiv API for the given query.
        """
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance"
        }

        # --- ADD THIS HEADER ---
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        # --- END ADDITION ---

        # --- UPDATE THIS LINE ---
        response = requests.get(self.base_url, params=params, headers=headers)
        # --- END UPDATE ---
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch papers: {response.status_code} - {response.text}")

        papers = self._parse_response(response.text)
        if years:
            years_str = set(str(y) for y in years)
            papers = [p for p in papers if p['published'][:4] in years_str]
        return papers

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
                'title': entry.find('atom:title', ns).text.strip(),
                'summary': entry.find('atom:summary', ns).text.strip(),
                'published': entry.find('atom:published', ns).text,
                'link': entry.find('atom:id', ns).text,
                'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            }
            papers.append(paper)
        return papers