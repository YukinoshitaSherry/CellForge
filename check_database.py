#!/usr/bin/env python3
"""
Check database content and data volume
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cellforge', 'Task_Analysis'))

from search import HybridSearcher

def check_database_content():
    """Check the content of databases"""
    print("🔍 Checking Database Content")
    print("=" * 50)
    
    searcher = HybridSearcher()
    
    # Check main database
    if searcher.qdrant_main:
        try:
            collections = searcher.qdrant_main.get_collections()
            print(f"📊 Main Database Collections:")
            for col in collections.collections:
                try:
                    count = searcher.qdrant_main.count(collection_name=col.name).count
                    print(f"  - {col.name}: {count} documents")
                except Exception as e:
                    print(f"  - {col.name}: Error getting count - {e}")
        except Exception as e:
            print(f"❌ Main database connection failed: {e}")
    
    # Check tmp database
    if searcher.qdrant_tmp:
        try:
            collections = searcher.qdrant_tmp.get_collections()
            print(f"\n📊 Tmp Database Collections:")
            for col in collections.collections:
                try:
                    count = searcher.qdrant_tmp.count(collection_name=col.name).count
                    print(f"  - {col.name}: {count} documents")
                except Exception as e:
                    print(f"  - {col.name}: Error getting count - {e}")
        except Exception as e:
            print(f"❌ Tmp database connection failed: {e}")
    
    # Test a simple search to see what we get
    print(f"\n🧪 Testing Search with 'single cell'")
    try:
        results = searcher.search("single cell", limit=10)
        print(f"📄 Search returned {len(results)} results")
        
        if results:
            print("📝 Sample results:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result.title[:60]}... (score: {result.score:.3f})")
                print(f"     Source: {result.source}")
                print(f"     Snippet: {result.snippet[:100]}...")
                print()
    except Exception as e:
        print(f"❌ Search test failed: {e}")

if __name__ == "__main__":
    check_database_content() 