#!/usr/bin/env python3
"""
Add sample data to increase search results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cellforge', 'Task_Analysis'))

from search import HybridSearcher
import uuid

def add_sample_data():
    """Add more sample data to increase search results"""
    print("📝 Adding Sample Data to Database")
    print("=" * 50)
    
    searcher = HybridSearcher()
    
    # Sample papers for single cell perturbation analysis
    sample_papers = [
        {
            "title": "Single-cell RNA sequencing reveals cell type-specific responses to CRISPR perturbations",
            "content": "This study demonstrates the use of single-cell RNA sequencing to analyze gene expression changes following CRISPR-mediated gene perturbations in different cell types. The authors developed a computational pipeline for analyzing perturbation effects at single-cell resolution.",
            "keywords": ["single-cell", "CRISPR", "perturbation", "RNA-seq", "gene expression"]
        },
        {
            "title": "Predictive modeling of gene expression responses to genetic perturbations",
            "content": "We present a deep learning model that predicts gene expression changes in response to genetic perturbations. The model uses attention mechanisms to capture regulatory relationships between genes.",
            "keywords": ["predictive model", "gene expression", "genetic perturbation", "deep learning", "attention"]
        },
        {
            "title": "Experimental design for single-cell perturbation studies",
            "content": "This paper provides guidelines for designing experiments to study cellular responses to perturbations at single-cell resolution, including considerations for sample size, controls, and analysis methods.",
            "keywords": ["experimental design", "single-cell", "perturbation", "sample size", "controls"]
        },
        {
            "title": "CRISPR-Cas9 mediated gene editing in single cells",
            "content": "We describe a protocol for CRISPR-Cas9 mediated gene editing in single cells, including guide RNA design, delivery methods, and validation strategies.",
            "keywords": ["CRISPR-Cas9", "gene editing", "single cells", "guide RNA", "validation"]
        },
        {
            "title": "Deep learning approaches for single-cell transcriptomics",
            "content": "This review covers recent advances in deep learning methods for analyzing single-cell transcriptomics data, including dimensionality reduction, clustering, and trajectory inference.",
            "keywords": ["deep learning", "single-cell", "transcriptomics", "dimensionality reduction", "clustering"]
        },
        {
            "title": "Machine learning for predicting cellular responses to perturbations",
            "content": "We develop machine learning models to predict how cells respond to various perturbations, including drug treatments and genetic modifications.",
            "keywords": ["machine learning", "cellular responses", "perturbations", "drug treatments", "genetic modifications"]
        },
        {
            "title": "Single-cell perturbation analysis using graph neural networks",
            "content": "We propose a graph neural network approach for analyzing single-cell perturbation data, capturing complex interactions between genes and cell states.",
            "keywords": ["graph neural networks", "single-cell", "perturbation analysis", "gene interactions", "cell states"]
        },
        {
            "title": "Transcriptional regulation in response to cellular perturbations",
            "content": "This study investigates how transcriptional regulation changes in response to various cellular perturbations, revealing key regulatory networks.",
            "keywords": ["transcriptional regulation", "cellular perturbations", "regulatory networks", "gene expression"]
        },
        {
            "title": "High-throughput screening of genetic perturbations",
            "content": "We describe a high-throughput screening platform for testing genetic perturbations in single cells, enabling large-scale functional genomics studies.",
            "keywords": ["high-throughput screening", "genetic perturbations", "single cells", "functional genomics"]
        },
        {
            "title": "Computational methods for single-cell perturbation analysis",
            "content": "This review covers computational methods and tools for analyzing single-cell perturbation data, including preprocessing, normalization, and statistical analysis.",
            "keywords": ["computational methods", "single-cell", "perturbation analysis", "preprocessing", "normalization"]
        }
    ]
    
    # Add to tmp database (which seems to have the CellForge collection)
    if searcher.qdrant_tmp:
        try:
            print(f"📤 Adding {len(sample_papers)} sample papers to tmp database...")
            
            for i, paper in enumerate(sample_papers):
                # Create embedding
                text_content = f"{paper['title']} {paper['content']}"
                embedding = searcher.encoder.encode(text_content)
                
                # Add to CellForge collection
                searcher.qdrant_tmp.upsert(
                    collection_name="CellForge",
                    points=[{
                        "id": hash(f"sample_paper_{i}_{uuid.uuid4()}") % 1000000,
                        "vector": embedding.tolist(),
                        "payload": {
                            "text": text_content,
                            "source": paper["title"],
                            "keywords": paper["keywords"],
                            "type": "sample_paper"
                        }
                    }]
                )
            
            print(f"✅ Successfully added {len(sample_papers)} sample papers")
            
            # Check the count
            count = searcher.qdrant_tmp.count(collection_name="CellForge").count
            print(f"📊 Total documents in CellForge collection: {count}")
            
        except Exception as e:
            print(f"❌ Failed to add sample data: {e}")
    else:
        print("❌ Tmp database not available")

if __name__ == "__main__":
    add_sample_data() 