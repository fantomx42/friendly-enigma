import requests
import re
from typing import List, Dict

class WebSearch:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Performs a search using DuckDuckGo HTML (no API key required).
        """
        print(f"[Web] Searching for: {query}")
        try:
            url = f"https://html.duckduckgo.com/html/?q={query}"
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            
            # Simple regex parsing to avoid heavy BS4 dependency if possible
            # But let's assume we want to be robust. 
            # If BS4 isn't there, we'll use regex.
            
            results = []
            
            # Regex fallback for links
            link_pattern = r'<a class="result__a" href="([^"]+)">([^<]+)</a>'
            matches = re.findall(link_pattern, resp.text)
            
            for href, title in matches[:num_results]:
                results.append({
                    "title": title,
                    "url": href,
                    "snippet": "..." # Snippet extraction is harder with regex on DDG HTML
                })
                
            # If no results (maybe structure changed), return error
            if not results:
                return [{"title": "No results found", "url": "", "snippet": "Try a different query."}]
                
            return results

        except Exception as e:
            return [{"title": "Error", "url": "", "snippet": str(e)}]

    def fetch_page(self, url: str) -> str:
        """
        Fetches and summarizes a webpage.
        """
        print(f"[Web] Fetching: {url}")
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            
            # naive text extraction
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:3000] # Return first 3000 chars
        except Exception as e:
            return f"Error fetching page: {e}"

# Global instance
web = WebSearch()
