"""
Transcript Indexer for Parenting AI SMS System
Manual script to index transcripts into vector store
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parenting_ai.utils import load_config, setup_logging
from parenting_ai.rag_engine import RAGEngine

def main():
    """Main function for transcript indexing"""
    parser = argparse.ArgumentParser(description="Index transcripts for Parenting AI")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--dir", "--directory", dest="transcripts_dir", help="Directory containing transcript JSON files")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", action="store_true", help="Test search after indexing")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Setup logging
    logger = setup_logging(config)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine(config, transcripts_dir=args.transcripts_dir)
        
        # Rebuild index if requested
        if args.rebuild:
            logger.info("Rebuilding index from scratch...")
            rag_engine.rebuild_index()
        else:
            logger.info("Index is ready")
        
        # Get index statistics
        stats = rag_engine.get_index_stats()
        logger.info(f"Index statistics: {stats}")
        
        # Test search if requested
        if args.test:
            logger.info("Testing search functionality...")
            test_result = rag_engine.test_search()
            logger.info(f"Test search result: {test_result}")
        
        logger.info("Transcript indexing complete")
        return 0
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())



