from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .utils import TextProcessor
import requests
import os
from datetime import datetime
import json
import hashlib
import PyPDF2
import time
import logging

# log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result data structure"""
    title: str
    content: str
    source: str
    url: str
    relevance_score: float
    publication_date: Optional[str]
    citations: Optional[int]
    metadata: Dict[str, Any]
    stars: Optional[int] = None
    forks: Optional[int] = None
    language: Optional[str] = None
    last_updated: Optional[str] = None

class HybridSearcher:
    """
    Hybrid searcher implementing alternating BFS-DFS search strategy
    """
    def __init__(self, qdrant_url: str = "localhost", qdrant_port: int = 6333):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant_client = QdrantClient(url=qdrant_url, port=qdrant_port)
        self.text_processor = TextProcessor()
        
        # API tokens
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.pubmed_api_key = os.getenv("PUBMED_API_KEY")
        
        # API headers
        if self.github_token and self.github_token != "your_github_token_here":
            self.github_headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
        else:
            self.github_headers = None
            logger.info("GitHub token not configured, GitHub search will be disabled")
        
        self.pubmed_headers = {
            "api-key": self.pubmed_api_key
        }
        
        # Search parameters
        self.max_layers = 10  # L_max = 10
        self.query_overlap_threshold = 0.8  # τ = 0.8
        self.relevance_threshold = 0.5  # ε = 0.5
        
        # retry
        self.max_retries = 3
        self.retry_delay = 1  # second
        
        # Initialize static corpus
        self._initialize_static_corpus()
    
    def _initialize_static_corpus(self) -> None:
        """Initialize static corpus of specialized papers and decision support data"""
        # Load static corpus from JSON
        corpus_path = os.path.join(os.path.dirname(__file__), "data", "static_corpus.json")
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                self.static_corpus = json.load(f)
                
            # Index static corpus data
            self._index_static_corpus()
        else:
            print("Warning: Static corpus not found")
            self.static_corpus = {"papers": [], "experimental_designs": [], "evaluation_frameworks": [], "implementation_guides": []}
    
    def _index_static_corpus(self) -> None:
        """Index static corpus data into vector database"""
        try:
            # Index papers
            for paper in self.static_corpus.get("papers", []):
                self._index_static_paper(paper)
            
            # Index experimental designs
            for design in self.static_corpus.get("experimental_designs", []):
                self._index_static_experimental_design(design)
            
            # Index evaluation frameworks
            for framework in self.static_corpus.get("evaluation_frameworks", []):
                self._index_static_evaluation_framework(framework)
            
            # Index implementation guides
            for guide in self.static_corpus.get("implementation_guides", []):
                self._index_static_implementation_guide(guide)
                
        except Exception as e:
            logger.error(f"Error indexing static corpus: {str(e)}")
    
    def _index_static_paper(self, paper: Dict[str, Any]) -> None:
        """Index static paper data"""
        text_content = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        text_embedding = self.encoder.encode(text_content)
        
        point = models.PointStruct(
            id=hash(paper.get("title", "")),
            vector=text_embedding.tolist(),
            payload={
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "publication_date": paper.get("publication_date"),
                "journal": paper.get("journal", ""),
                "abstract": paper.get("abstract", ""),
                "content": text_content,
                "source": "Static Corpus",
                "url": paper.get("url", ""),
                "citations": paper.get("citations"),
                "metadata": paper.get("metadata", {}),
                "type": "paper"
            }
        )
        
        self.qdrant_client.upsert(
            collection_name="papers",
            points=[point]
        )
    
    def _index_static_experimental_design(self, design: Dict[str, Any]) -> None:
        """Index static experimental design data"""
        text_content = f"{design.get('title', '')} {design.get('content', '')}"
        text_embedding = self.encoder.encode(text_content)
        
        point = models.PointStruct(
            id=hash(design.get("title", "")),
            vector=text_embedding.tolist(),
            payload={
                "title": design.get("title", ""),
                "content": text_content,
                "source": "Static Corpus",
                "url": "",
                "metadata": design.get("metadata", {}),
                "type": "experimental_design"
            }
        )
        
        self.qdrant_client.upsert(
            collection_name="papers",
            points=[point]
        )
    
    def _index_static_evaluation_framework(self, framework: Dict[str, Any]) -> None:
        """Index static evaluation framework data"""
        text_content = f"{framework.get('title', '')} {framework.get('content', '')}"
        text_embedding = self.encoder.encode(text_content)
        
        point = models.PointStruct(
            id=hash(framework.get("title", "")),
            vector=text_embedding.tolist(),
            payload={
                "title": framework.get("title", ""),
                "content": text_content,
                "source": "Static Corpus",
                "url": "",
                "metadata": framework.get("metadata", {}),
                "type": "evaluation_framework"
            }
        )
        
        self.qdrant_client.upsert(
            collection_name="papers",
            points=[point]
        )
    
    def _index_static_implementation_guide(self, guide: Dict[str, Any]) -> None:
        """Index static implementation guide data"""
        text_content = f"{guide.get('title', '')} {guide.get('content', '')}"
        text_embedding = self.encoder.encode(text_content)
        
        point = models.PointStruct(
            id=hash(guide.get("title", "")),
            vector=text_embedding.tolist(),
            payload={
                "title": guide.get("title", ""),
                "content": text_content,
                "source": "Static Corpus",
                "url": "",
                "metadata": guide.get("metadata", {}),
                "type": "implementation_guide"
            }
        )
        
        self.qdrant_client.upsert(
            collection_name="papers",
            points=[point]
        )
    
    def _search_pubmed(self, keywords: List[str]) -> List[SearchResult]:
        """Search PubMed for relevant papers using NCBI E-utilities"""
        # Check if PubMed is configured
        pubmed_api_key = os.getenv("PUBMED_API_KEY")
        pubmed_tool = os.getenv("PUBMED_TOOL")
        pubmed_email = os.getenv("PUBMED_EMAIL")
        
        # If PubMed is not configured, return empty results
        if not pubmed_tool or not pubmed_email:
            logger.info("PubMed not configured, skipping PubMed search")
            return []
        
        query = " ".join(keywords)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": 10,
            "sort": "relevance",
            "retmode": "json",
            "tool": pubmed_tool,
            "email": pubmed_email
        }
        
        # Add API key if available
        if pubmed_api_key and pubmed_api_key != "your_pubmed_api_key_here":
            params["api_key"] = pubmed_api_key
        
        try:
            # Use retry mechanism with proper rate limiting
            response = self._make_request_with_retry(url, params=params)
            
            # Parse JSON response
            data = response.json()
            if "esearchresult" in data and "idlist" in data["esearchresult"]:
                pubmed_ids = data["esearchresult"]["idlist"]
            else:
                logger.warning("No PubMed IDs found in response")
                return []
            
            # Get paper details with rate limiting
            results = []
            for i, pmid in enumerate(pubmed_ids):
                # Rate limiting: max 3 requests per second without API key, 10 with API key
                if i > 0:
                    time.sleep(0.1 if pubmed_api_key else 0.34)
                
                paper_details = self._get_pubmed_paper_details(pmid)
                if paper_details:
                    results.append(paper_details)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []
    
    def _get_pubmed_paper_details(self, pmid: str) -> Optional[SearchResult]:
        """Get detailed information for a PubMed paper using NCBI E-utilities"""
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        # Get PubMed configuration from environment
        pubmed_api_key = os.getenv("PUBMED_API_KEY")
        pubmed_tool = os.getenv("PUBMED_TOOL", "BioForge")
        pubmed_email = os.getenv("PUBMED_EMAIL")
        
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "tool": pubmed_tool,
            "email": pubmed_email
        }
        
        # Add API key if available
        if pubmed_api_key and pubmed_api_key != "your_pubmed_api_key_here":
            params["api_key"] = pubmed_api_key
        
        try:
            # Use retry mechanism
            response = self._make_request_with_retry(url, params=params)
            
            # Parse XML response
            xml_content = response.text
            
            # Extract title
            title_start = xml_content.find("<ArticleTitle>")
            title_end = xml_content.find("</ArticleTitle>")
            title = xml_content[title_start + 14:title_end] if title_start != -1 and title_end != -1 else "Unknown Title"
            
            # Extract abstract
            abstract_start = xml_content.find("<AbstractText>")
            abstract_end = xml_content.find("</AbstractText>")
            abstract = xml_content[abstract_start + 14:abstract_end] if abstract_start != -1 and abstract_end != -1 else "No abstract available"
            
            # Extract publication date
            pub_date_start = xml_content.find("<PubDate>")
            pub_date_end = xml_content.find("</PubDate>")
            pub_date = xml_content[pub_date_start + 9:pub_date_end] if pub_date_start != -1 and pub_date_end != -1 else None
            
            # Extract authors
            authors = []
            author_sections = xml_content.split("<Author>")
            for section in author_sections[1:]:  # Skip first empty section
                last_name_start = section.find("<LastName>")
                last_name_end = section.find("</LastName>")
                first_name_start = section.find("<ForeName>")
                first_name_end = section.find("</ForeName>")
                
                if last_name_start != -1 and last_name_end != -1:
                    last_name = section[last_name_start + 10:last_name_end]
                    first_name = section[first_name_start + 10:first_name_end] if first_name_start != -1 and first_name_end != -1 else ""
                    authors.append(f"{first_name} {last_name}".strip())
            
            # Extract journal
            journal_start = xml_content.find("<Journal>")
            journal_end = xml_content.find("</Journal>")
            journal = xml_content[journal_start + 9:journal_end] if journal_start != -1 and journal_end != -1 else "Unknown Journal"
            
            return SearchResult(
                title=title,
                content=abstract,
                source="pubmed",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                relevance_score=0.0,  # Will be updated by search
                publication_date=pub_date,
                citations=None,  # Would need additional API call
                metadata={
                    "pmid": pmid,
                    "authors": authors,
                    "journal": journal
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting PubMed paper details: {str(e)}")
            return None
    
    def search(self, keywords: List[str]) -> Dict[str, List[SearchResult]]:
        """
        Execute hybrid BFS-DFS search
        
        Args:
            keywords: List of search keywords
            
        Returns:
            Dictionary containing search results from different sources
        """
        # Initialize search parameters
        visited = set()
        query_history = []
        all_results = {
            "papers": [],
            "code": [],
            "github": [],
            "pubmed": []
        }
        
        # Initial query embedding
        query_embedding = self.encoder.encode(" ".join(keywords))
        
        # Execute BFS-DFS search
        for layer in range(self.max_layers):
            # Alternate between BFS and DFS
            if layer % 2 == 0:
                layer_results = self._bfs_search(query_embedding, visited, layer)
            else:
                layer_results = self._dfs_search(query_embedding, visited, layer)
            
            # Add PubMed results
            layer_results["pubmed"] = self._search_pubmed(keywords)
            
            # Merge results
            for source in all_results:
                all_results[source].extend(layer_results[source])
            
            # Check stopping condition
            if self._should_stop(layer_results, query_history):
                break
            
            # Update query for next layer
            new_query = self._update_query(layer_results)
            query_history.append(new_query)
            query_embedding = self.encoder.encode(new_query)
        
        # Sort results by relevance score
        for source in all_results:
            all_results[source].sort(key=lambda x: x.relevance_score, reverse=True)
            all_results[source] = all_results[source][:5]  # Keep top 5 results
        
        return all_results
    
    def _search_papers(self, keywords: List[str]) -> List[SearchResult]:
        query_vector = self.encoder.encode(" ".join(keywords))
        search_result = self.qdrant_client.search(
            collection_name="papers",
            query_vector=query_vector,
            limit=5
        )
        
        results = []
        for hit in search_result:
            results.append(SearchResult(
                title=hit.payload.get("title", ""),
                content=hit.payload.get("content", ""),
                source="papers",
                url=hit.payload.get("url", ""),
                relevance_score=hit.score,
                publication_date=hit.payload.get("publication_date"),
                citations=hit.payload.get("citations"),
                metadata=hit.payload.get("metadata", {})
            ))
        return results
    
    def _search_code(self, keywords: List[str]) -> List[SearchResult]:
        query_vector = self.encoder.encode(" ".join(keywords))
        search_result = self.qdrant_client.search(
            collection_name="code",
            query_vector=query_vector,
            limit=5
        )
        
        results = []
        for hit in search_result:
            results.append(SearchResult(
                title=hit.payload.get("title", ""),
                content=hit.payload.get("content", ""),
                source="code",
                url=hit.payload.get("url", ""),
                relevance_score=hit.score,
                publication_date=None,
                citations=None,
                metadata=hit.payload.get("metadata", {})
            ))
        return results
    
    def _search_github(self, keywords: List[str]) -> List[SearchResult]:
        """Search GitHub repositories for relevant code"""
        # Check if GitHub is configured
        if not self.github_headers:
            logger.info("GitHub not configured, skipping GitHub search")
            return []
        
        query = " ".join(keywords)
        url = f"https://api.github.com/search/repositories"
        params = {
            "q": f"{query} language:python stars:>100",
            "sort": "stars",
            "order": "desc",
            "per_page": 10
        }
        
        try:
            response = requests.get(url, headers=self.github_headers, params=params)
            response.raise_for_status()
            repos = response.json()["items"]
            
            results = []
            for repo in repos:
                # Get repository details
                repo_url = repo["html_url"]
                readme_content = self._get_repo_readme(repo["owner"]["login"], repo["name"])
                
                # Create search result
                result = SearchResult(
                    title=repo["full_name"],
                    content=readme_content or repo["description"],
                    source="github",
                    url=repo_url,
                    relevance_score=self._calculate_relevance_score(repo, keywords),
                    publication_date=None,
                    citations=None,
                    metadata={
                        "language": repo["language"],
                        "topics": repo["topics"],
                        "size": repo["size"],
                        "open_issues": repo["open_issues"]
                    },
                    stars=repo["stargazers_count"],
                    forks=repo["forks_count"],
                    language=repo["language"],
                    last_updated=repo["updated_at"]
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching GitHub: {str(e)}")
            return []
    
    def _get_repo_readme(self, owner: str, repo: str) -> Optional[str]:
        """Get repository README content"""
        if not self.github_headers:
            return None
            
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/readme"
            response = requests.get(url, headers=self.github_headers)
            response.raise_for_status()
            return response.json().get("content", "")
        except:
            return None
    
    def _calculate_relevance_score(self, repo: Dict[str, Any], keywords: List[str]) -> float:
        """Calculate relevance score for GitHub repository"""
        score = 0.0
        
        # Base score from stars
        score += min(repo["stargazers_count"] / 1000, 1.0) * 0.3
        
        # Score from description and topics
        text = f"{repo['description']} {' '.join(repo['topics'])}"
        for keyword in keywords:
            if keyword.lower() in text.lower():
                score += 0.1
        
        # Score from language match
        if repo["language"] and repo["language"].lower() in ["python", "jupyter notebook"]:
            score += 0.2
            
        return min(score, 1.0)
    
    def _bfs_search(self, 
                    query_embedding: np.ndarray,
                    visited: set,
                    layer: int) -> Dict[str, List[SearchResult]]:
        """
        Execute breadth-first search
        
        Args:
            query_embedding: Query embedding vector
            visited: Set of visited documents
            layer: Current layer number
            
        Returns:
            Dictionary containing search results
        """
        results = {
            "papers": [],
            "code": [],
            "github": []
        }
        
        # Search papers
        paper_results = self.qdrant_client.search(
            collection_name="papers",
            query_vector=query_embedding.tolist(),
            limit=5
        )
        
        # Process paper results
        for result in paper_results:
            if result.score >= self.relevance_threshold and result.payload["url"] not in visited:
                visited.add(result.payload["url"])
                search_result = SearchResult(
                    title=result.payload.get("title", ""),
                    content=result.payload.get("content", ""),
                    source=result.payload.get("source", ""),
                    url=result.payload.get("url", ""),
                    relevance_score=result.score,
                    publication_date=result.payload.get("publication_date"),
                    citations=result.payload.get("citations"),
                    metadata=result.payload.get("metadata", {})
                )
                results["papers"].append(search_result)
        
        # Search code
        code_results = self.qdrant_client.search(
            collection_name="code",
            query_vector=query_embedding.tolist(),
            limit=5
        )
        
        # Process code results
        for result in code_results:
            if result.score >= self.relevance_threshold and result.payload["url"] not in visited:
                visited.add(result.payload["url"])
                search_result = SearchResult(
                    title=result.payload.get("title", ""),
                    content=result.payload.get("content", ""),
                    source=result.payload.get("source", ""),
                    url=result.payload.get("url", ""),
                    relevance_score=result.score,
                    publication_date=None,
                    citations=None,
                    metadata=result.payload.get("metadata", {})
                )
                results["code"].append(search_result)
        
        return results
    
    def _dfs_search(self,
                    query_embedding: np.ndarray,
                    visited: set,
                    layer: int) -> Dict[str, List[SearchResult]]:
        """
        Execute depth-first search
        
        Args:
            query_embedding: Query embedding vector
            visited: Set of visited documents
            layer: Current layer number
            
        Returns:
            Dictionary containing search results
        """
        results = {
            "papers": [],
            "code": [],
            "github": []
        }
        
        # Get initial results
        initial_results = self._bfs_search(query_embedding, visited, layer)
        
        # Follow highest-scoring paths
        for paper in initial_results["papers"]:
            if paper.relevance_score >= self.relevance_threshold:
                # Get related papers
                related_papers = self._get_related_papers(paper)
                
                # Process related papers
                for related_paper in related_papers:
                    if related_paper.url not in visited:
                        visited.add(related_paper.url)
                        results["papers"].append(related_paper)
        
        return results
    
    def _get_related_papers(self, paper: SearchResult) -> List[SearchResult]:
        """
        Get related papers based on content similarity
        
        Args:
            paper: Source paper
            
        Returns:
            List of related papers
        """
        results = []
        
        # Encode paper content
        paper_embedding = self.encoder.encode(paper.content)
        
        # Search for related papers
        search_results = self.qdrant_client.search(
            collection_name="papers",
            query_vector=paper_embedding.tolist(),
            limit=5
        )
        
        # Process results
        for result in search_results:
            if result.score >= self.relevance_threshold:
                search_result = SearchResult(
                    title=result.payload.get("title", ""),
                    content=result.payload.get("content", ""),
                    source=result.payload.get("source", ""),
                    url=result.payload.get("url", ""),
                    relevance_score=result.score,
                    publication_date=result.payload.get("publication_date"),
                    citations=result.payload.get("citations"),
                    metadata=result.payload.get("metadata", {})
                )
                results.append(search_result)
        
        return results
    
    def _should_stop(self, 
                    layer_results: Dict[str, List[SearchResult]],
                    query_history: List[str]) -> bool:
        """
        Check if search should stop based on stopping conditions
        
        Args:
            layer_results: Results from current layer
            query_history: History of queries
            
        Returns:
            True if search should stop, False otherwise
        """
        # Check if maximum layers reached
        if len(query_history) >= self.max_layers:
            return True
        
        # Check if query overlap exceeds threshold
        if len(query_history) > 1:
            current_query = query_history[-1]
            previous_query = query_history[-2]
            overlap = self.text_processor.calculate_text_similarity(
                current_query,
                previous_query
            )
            if overlap >= self.query_overlap_threshold:
                return True
        
        # Check if relevance scores are below threshold
        if not any(
            result.relevance_score >= self.relevance_threshold
            for results in layer_results.values()
            for result in results
        ):
            return True
        
        return False
    
    def _update_query(self, 
                     layer_results: Dict[str, List[SearchResult]]) -> str:
        """
        Update search query based on current results
        
        Args:
            layer_results: Results from current layer
            
        Returns:
            Updated query string
        """
        # Extract key information from results
        key_terms = []
        
        for results in layer_results.values():
            for result in results:
                if result.relevance_score >= self.relevance_threshold:
                    # Extract keywords from title and content
                    title_keywords = self.text_processor.extract_keywords(
                        result.title,
                        top_n=3
                    )
                    content_keywords = self.text_processor.extract_keywords(
                        result.content,
                        top_n=5
                    )
                    key_terms.extend(title_keywords + content_keywords)
        
        # Get most common terms
        if key_terms:
            from collections import Counter
            term_counter = Counter(key_terms)
            top_terms = [term for term, _ in term_counter.most_common(5)]
            return " ".join(top_terms)
        
        return ""

    def _make_request_with_retry(self, url: str, params: Dict = None, headers: Dict = None, 
                                method: str = "GET", json_data: Dict = None) -> requests.Response:
        """
        HTTP request with retry mechanism
        
        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            method: Request method
            json_data: JSON data
            
        Returns:
            requests.Response object
        """
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, params=params, headers=headers, timeout=10)
                elif method.upper() == "POST":
                    response = requests.post(url, params=params, headers=headers, json=json_data, timeout=10)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts failed for {url}")
                    raise

    def search_decision_support(self, task_description: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for decision support information specifically
        
        Args:
            task_description: Task description
            dataset_info: Dataset information
            
        Returns:
            Decision support information
        """
        # Build decision support queries
        decision_queries = [
            f"experimental design {task_description}",
            f"evaluation framework {task_description}",
            f"implementation guide {task_description}",
            f"model recommendation {task_description}",
            f"data requirements {task_description}"
        ]
        
        decision_results = {}
        
        for query in decision_queries:
            query_vector = self.encoder.encode(query)
            search_result = self.qdrant_client.search(
                collection_name="papers",
                query_vector=query_vector,
                limit=3,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type",
                            match=models.MatchValue(value="experimental_design")
                        )
                    ]
                )
            )
            
            if search_result:
                decision_results[query] = [
                    {
                        "title": hit.payload.get("title", ""),
                        "content": hit.payload.get("content", ""),
                        "metadata": hit.payload.get("metadata", {}),
                        "score": hit.score
                    }
                    for hit in search_result
                ]
        
        return decision_results
    
    def get_decision_recommendations(self, task_description: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive decision recommendations
        
        Args:
            task_description: Task description
            dataset_info: Dataset information
            
        Returns:
            Comprehensive decision recommendations
        """
        # Search for related papers and code
        search_results = self.search(task_description.split())
        
        # Extract decision support information
        decision_support = self._extract_comprehensive_decision_support(search_results, dataset_info)
        
        # Generate recommendations
        recommendations = {
            "model_selection": self._generate_model_selection_recommendations(decision_support),
            "evaluation_strategy": self._generate_evaluation_strategy_recommendations(decision_support),
            "data_preparation": self._generate_data_preparation_recommendations(decision_support, dataset_info),
            "implementation_plan": self._generate_implementation_plan_recommendations(decision_support),
            "risk_assessment": self._generate_risk_assessment_recommendations(decision_support)
        }
        
        return recommendations
    
    def _extract_comprehensive_decision_support(self, search_results: Dict[str, List[SearchResult]], 
                                              dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive decision support information"""
        decision_support = {
            "models": [],
            "metrics": [],
            "data_requirements": [],
            "complexity_levels": [],
            "interpretability_levels": []
        }
        
        for source, results in search_results.items():
            for result in results:
                if hasattr(result, 'metadata') and result.metadata:
                    support = result.metadata.get("decision_support", {})
                    if support:
                        decision_support["models"].extend(support.get("model_recommendations", []))
                        decision_support["metrics"].extend(support.get("evaluation_metrics", []))
                        decision_support["data_requirements"].extend(support.get("data_requirements", []))
                        decision_support["complexity_levels"].append(support.get("implementation_complexity", "unknown"))
                        decision_support["interpretability_levels"].append(support.get("biological_interpretability", "unknown"))
        
        return decision_support
    
    def _generate_model_selection_recommendations(self, decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model selection recommendations"""
        model_counts = {}
        for model in decision_support["models"]:
            model_counts[model] = model_counts.get(model, 0) + 1
        
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "top_recommendations": [model for model, count in sorted_models[:3]],
            "all_recommendations": [model for model, count in sorted_models],
            "confidence": min(len(sorted_models) / 5, 1.0)
        }
    
    def _generate_evaluation_strategy_recommendations(self, decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation strategy recommendations"""
        metric_counts = {}
        for metric in decision_support["metrics"]:
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        sorted_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_metrics": [metric for metric, count in sorted_metrics[:3]],
            "secondary_metrics": [metric for metric, count in sorted_metrics[3:6]],
            "validation_approach": "Independent test set with biological validation recommended"
        }
    
    def _generate_data_preparation_recommendations(self, decision_support: Dict[str, Any], 
                                                 dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data preparation recommendations"""
        required_data = set(decision_support["data_requirements"])
        available_data = set(dataset_info.get("characteristics", []))
        missing_data = required_data - available_data
        
        return {
            "required_preprocessing": list(required_data),
            "missing_requirements": list(missing_data),
            "compatibility_score": len(available_data) / max(len(required_data), 1),
            "recommendations": [
                "Ensure data quality control",
                "Apply appropriate normalization",
                "Handle batch effects if present"
            ]
        }
    
    def _generate_implementation_plan_recommendations(self, decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implementation plan recommendations"""
        complexity_levels = decision_support["complexity_levels"]
        complexity_scores = {"very_high": 4, "high": 3, "moderate": 2, "low": 1, "unknown": 2}
        
        avg_complexity = sum(complexity_scores.get(level, 2) for level in complexity_levels) / max(len(complexity_levels), 1)
        
        if avg_complexity >= 3.5:
            timeline = "3-6 months"
            team_size = "3-5 researchers"
            resources = "High-performance computing cluster"
        elif avg_complexity >= 2.5:
            timeline = "2-4 months"
            team_size = "2-3 researchers"
            resources = "GPU workstation"
        else:
            timeline = "1-2 months"
            team_size = "1-2 researchers"
            resources = "Standard workstation"
        
        return {
            "estimated_timeline": timeline,
            "recommended_team_size": team_size,
            "resource_requirements": resources,
            "complexity_level": "high" if avg_complexity >= 3 else "moderate" if avg_complexity >= 2 else "low"
        }
    
    def _generate_risk_assessment_recommendations(self, decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment recommendations"""
        interpretability_levels = decision_support["interpretability_levels"]
        interpretability_scores = {"very_high": 4, "high": 3, "moderate": 2, "low": 1, "unknown": 2}
        
        avg_interpretability = sum(interpretability_scores.get(level, 2) for level in interpretability_levels) / max(len(interpretability_levels), 1)
        
        risks = []
        if avg_interpretability <= 2:
            risks.append("Low biological interpretability may limit clinical translation")
        if len(decision_support["models"]) == 0:
            risks.append("Limited model recommendations available")
        if len(decision_support["metrics"]) == 0:
            risks.append("No specific evaluation metrics identified")
        
        return {
            "identified_risks": risks,
            "interpretability_concern": avg_interpretability <= 2,
            "mitigation_strategies": [
                "Include biological validation experiments",
                "Use interpretable model architectures",
                "Implement comprehensive evaluation framework"
            ]
        }

def alternating_search(query, max_layers=5, threshold=0.7, epsilon=0.3):
    results = []
    current_query = query
    visited = set()
    
    for layer in range(max_layers):
        if layer % 2 == 0:
            new_results = breadth_first_search(current_query, visited)
        else:
            new_results = depth_first_search(current_query, visited)
            
        if not new_results:
            break
            
        results.extend(new_results)
        visited.update(r['id'] for r in new_results)
        
        current_query = update_query(current_query, new_results)
        
        if check_termination(current_query, results, threshold, epsilon):
            break
            
    return results

def enhanced_retrieval(query, cache_dir='.cache'):
    cache_key = hashlib.md5(query.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
            
    results = {
        'papers': search_pubmed(query),
        'code': search_github(query),
        'web': search_serpapi(query),
        'local': search_local_pdfs(query)
    }
    
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(results, f)
        
    return results

def format_retrieval_results(results):
    formatted = {
        'papers': [{
            'title': p['title'],
            'authors': p['authors'],
            'year': p['year'],
            'url': p['url'],
            'abstract': p['abstract'],
            'score': p['score']
        } for p in results['papers']],
        
        'code': [{
            'repo': c['repo'],
            'description': c['description'],
            'stars': c['stars'],
            'url': c['url'],
            'score': c['score']
        } for c in results['code']],
        
        'web': [{
            'title': w['title'],
            'snippet': w['snippet'],
            'url': w['url'],
            'score': w['score']
        } for w in results['web']],
        
        'local': [{
            'title': l['title'],
            'content': l['content'],
            'path': l['path'],
            'score': l['score']
        } for l in results['local']]
    }
    
    return formatted

def update_query(current_query, new_results, alpha=0.7):
    if not new_results:
        return current_query
        
    new_embeddings = [r['embedding'] for r in new_results]
    avg_new_embedding = np.mean(new_embeddings, axis=0)
    
    return alpha * current_query + (1 - alpha) * avg_new_embedding

def check_termination(current_query, results, threshold, epsilon):
    if not results:
        return True
        
    overlap = calculate_overlap(current_query, results[-1]['embedding'])
    if overlap > threshold:
        return True
        
    max_score = max(r['score'] for r in results)
    if max_score < epsilon:
        return True
        
    return False

def calculate_overlap(query1, query2, threshold=0.8):
    similarity = cosine_similarity(query1, query2)
    return similarity > threshold

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_pubmed(query, max_results=50):
    """Search PubMed using NCBI E-utilities"""
    # Check if PubMed is configured
    pubmed_api_key = os.getenv("PUBMED_API_KEY")
    pubmed_tool = os.getenv("PUBMED_TOOL")
    pubmed_email = os.getenv("PUBMED_EMAIL")
    
    # If PubMed is not configured, return empty results
    if not pubmed_tool or not pubmed_email:
        logger.info("PubMed not configured, skipping PubMed search")
        return []
    
    try:
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'tool': pubmed_tool,
            'email': pubmed_email
        }
        
        # Add API key if available
        if pubmed_api_key and pubmed_api_key != "your_pubmed_api_key_here":
            params['api_key'] = pubmed_api_key
        
        response = requests.get(
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
            params=params
        )
        
        data = response.json()
        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            return data["esearchresult"]["idlist"]
        else:
            logger.warning("No PubMed IDs found in response")
            return []
            
    except Exception as e:
        logger.error(f"PubMed search failed: {str(e)}")
        return []

def search_github(query, max_results=50):
    """Search GitHub repositories"""
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        
        # If GitHub token is not configured, return empty results
        if not github_token or github_token == "your_github_token_here":
            logger.info("GitHub token not configured, skipping GitHub search")
            return []
            
        headers = {'Authorization': f'token {github_token}'}
            
        response = requests.get(
            'https://api.github.com/search/repositories',
            params={
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': max_results
            },
            headers=headers
        )
        return response.json()['items']
    except Exception as e:
        logger.error(f"GitHub search failed: {str(e)}")
        return []

def search_serpapi(query, max_results=50):
    try:
        serpapi_key = os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            logger.warning("SERPAPI_KEY not found in environment variables")
            return []
            
        response = requests.get(
            'https://serpapi.com/search',
            params={
                'q': query,
                'api_key': serpapi_key,
                'num': max_results
            }
        )
        return response.json()['organic_results']
    except Exception as e:
        logger.error(f"SERPAPI search failed: {str(e)}")
        return []

def search_local_pdfs(query, pdf_dir='./pdfs'):
    results = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.endswith('.pdf'):
                try:
                    path = os.path.join(root, file)
                    text = extract_pdf_text(path)
                    score = calculate_relevance(text, query)
                    if score > 0.5:
                        results.append({
                            'title': file,
                            'content': text,
                            'path': path,
                            'score': score
                        })
                except Exception as e:
                    logger.error(f"PDF processing failed for {file}: {str(e)}")
    return results

def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def calculate_relevance(text, query):
    # Use the global sentence transformer model
    global st_model
    if st_model is None:
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    text_embedding = st_model.encode(text)
    query_embedding = st_model.encode(query)
    return cosine_similarity(text_embedding, query_embedding) 