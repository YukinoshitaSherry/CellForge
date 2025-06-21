import os
from typing import Dict, Any
from .retriever import HybridRetriever
from .parser import PaperParser
from .indexer import VectorIndexer
from .search import HybridSearcher
from .utils import TextProcessor

def main():
    try:
        # Check environment variables
        qdrant_url = os.getenv("QDRANT_URL", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        # Check data directories
        papers_dir = "data/papers"
        code_dir = "data/code"
        
        if not os.path.exists(papers_dir):
            os.makedirs(papers_dir)
            print(f"Created papers directory: {papers_dir}")
            
        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
            print(f"Created code directory: {code_dir}")
        
        # Initialize components
        print("Initializing components...")
        retriever = HybridRetriever(qdrant_url, qdrant_port)
        parser = PaperParser()
        indexer = VectorIndexer(qdrant_url, qdrant_port)
        searcher = HybridSearcher(qdrant_url, qdrant_port)
        processor = TextProcessor()
        
        # Check Qdrant connection
        try:
            collections = indexer.qdrant_client.get_collections()
            print(f"Successfully connected to Qdrant at {qdrant_url}:{qdrant_port}")
            print(f"Available collections: {[col.name for col in collections.collections]}")
        except Exception as e:
            print(f"Error connecting to Qdrant: {str(e)}")
            print("Please make sure Qdrant is running and accessible")
            print("You can start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
            return
        
        # Index data
        print("\nIndexing papers...")
        indexer.index_papers(papers_dir)
        
        print("\nIndexing code...")
        indexer.index_code(code_dir)
        
        # Execute search
        print("\nExecuting search...")
        task_description = "Implement a deep learning model for gene expression prediction"
        dataset_info = {
            "name": "scRNA-seq",
            "modality": "gene expression",
            "task_type": "regression",
            "model_type": "neural network"
        }
        
        results = retriever.search(task_description, dataset_info)
        
        # Display results
        print("\nSearch Results:")
        for source, search_results in results.items():
            print(f"\nResults from {source}:")
            for result in search_results:
                print(f"\nTitle: {result.title}")
                print(f"Content: {result.content[:200]}...")
                print(f"Relevance Score: {result.relevance_score}")
                if source == "github":
                    print(f"Stars: {result.stars}")
                    print(f"Forks: {result.forks}")
                    print(f"Language: {result.language}")
                    print(f"Last Updated: {result.last_updated}")
                print(f"URL: {result.url}")
                print(f"Metadata: {result.metadata}")
                
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 