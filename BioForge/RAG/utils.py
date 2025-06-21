from typing import List, Dict, Any
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

class TextProcessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        self.technical_terms = [
            "transformer", "attention", "embedding", "encoder", "decoder",
            "neural network", "deep learning", "machine learning", "AI",
            "single-cell", "RNA-seq", "gene expression", "perturbation",
            "CRISPR", "knockout", "drug treatment", "cytokine"
        ]
        
        self.biological_terms = [
            "gene", "protein", "cell", "tissue", "organism", "pathway",
            "metabolism", "signaling", "transcription", "translation",
            "mutation", "variant", "allele", "genotype", "phenotype"
        ]
        
    def extract_keywords(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        return list(set(tokens))
    
    def extract_technical_terms(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_terms = []
        for term in self.technical_terms:
            if term.lower() in text_lower:
                found_terms.append(term)
        return found_terms
    
    def extract_biological_terms(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_terms = []
        for term in self.biological_terms:
            if term.lower() in text_lower:
                found_terms.append(term)
        return found_terms
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        return (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
    
    def extract_summary(self, text: str, max_length: int = 200) -> str:
        sentences = sent_tokenize(text)
        if not sentences:
            return ""
        
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        top_sentence_indices = sentence_scores.argsort()[-3:][::-1]
        top_sentence_indices.sort()
        
        summary = " ".join([sentences[i] for i in top_sentence_indices])
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        feature_names = self.vectorizer.get_feature_names_out()
        
        phrase_scores = []
        for i, sentence in enumerate(sentences):
            sentence_vector = tfidf_matrix[i].toarray()[0]
            top_indices = sentence_vector.argsort()[-top_n:][::-1]
            for idx in top_indices:
                if sentence_vector[idx] > 0:
                    phrase_scores.append((feature_names[idx], sentence_vector[idx]))
        
        phrase_counter = Counter([phrase for phrase, _ in phrase_scores])
        return [phrase for phrase, _ in phrase_counter.most_common(top_n)]
    
    def extract_technical_phrases(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        technical_phrases = []
        
        for sentence in sentences:
            technical_terms = self.extract_technical_terms(sentence)
            if technical_terms:
                technical_phrases.extend(technical_terms)
        
        return list(set(technical_phrases))
    
    def extract_biological_phrases(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        biological_phrases = []
        
        for sentence in sentences:
            biological_terms = self.extract_biological_terms(sentence)
            if biological_terms:
                biological_phrases.extend(biological_terms)
        
        return list(set(biological_phrases))

def view_pdfs_in_database(qdrant_url: str = "localhost", qdrant_port: int = 6333) -> List[Dict[str, Any]]:
    """
    View all PDFs stored in the vector database
    
    Args:
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
        
    Returns:
        List of PDF documents with metadata
    """
    try:
        client = QdrantClient(url=qdrant_url, port=qdrant_port)
        
        # Get all points from papers collection
        response = client.scroll(
            collection_name="papers",
            limit=100,  # Adjust as needed
            with_payload=True,
            with_vectors=False
        )
        
        pdfs = []
        for point in response[0]:
            payload = point.payload
            pdf_info = {
                "id": point.id,
                "title": payload.get("title", "Unknown"),
                "authors": payload.get("authors", []),
                "publication_date": payload.get("publication_date"),
                "journal": payload.get("journal", ""),
                "abstract": payload.get("abstract", "")[:200] + "..." if payload.get("abstract") else "",
                "source": payload.get("source", ""),
                "url": payload.get("url", ""),
                "citations": payload.get("citations"),
                "content_preview": payload.get("content", "")[:300] + "..." if payload.get("content") else ""
            }
            pdfs.append(pdf_info)
        
        return pdfs
        
    except Exception as e:
        print(f"Error accessing vector database: {str(e)}")
        return []

def search_pdfs_in_database(query: str, qdrant_url: str = "localhost", qdrant_port: int = 6333, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search PDFs in the vector database
    
    Args:
        query: Search query
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
        limit: Maximum number of results
        
    Returns:
        List of matching PDF documents
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Initialize encoder
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode query
        query_vector = encoder.encode(query).tolist()
        
        # Search in database
        client = QdrantClient(url=qdrant_url, port=qdrant_port)
        
        response = client.search(
            collection_name="papers",
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        results = []
        for point in response:
            payload = point.payload
            result = {
                "id": point.id,
                "score": point.score,
                "title": payload.get("title", "Unknown"),
                "authors": payload.get("authors", []),
                "publication_date": payload.get("publication_date"),
                "journal": payload.get("journal", ""),
                "abstract": payload.get("abstract", "")[:200] + "..." if payload.get("abstract") else "",
                "content_preview": payload.get("content", "")[:300] + "..." if payload.get("content") else ""
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error searching vector database: {str(e)}")
        return [] 