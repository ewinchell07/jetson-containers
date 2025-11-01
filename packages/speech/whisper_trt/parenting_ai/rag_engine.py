"""
RAG Engine for Parenting AI SMS System
Handles semantic search over transcript data using llama-index
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

try:
    from llama_index.core import (
        VectorStoreIndex, 
        SimpleDirectoryReader,
        StorageContext,
        load_index_from_storage
    )
    from llama_index.core.embeddings import MockEmbedding
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        HAS_HUGGINGFACE = True
    except ImportError:
        try:
            from llama_index.embeddings.sentence_transformer import SentenceTransformerEmbedding
            HAS_HUGGINGFACE = True
            HuggingFaceEmbedding = SentenceTransformerEmbedding
        except ImportError:
            HAS_HUGGINGFACE = False
            HuggingFaceEmbedding = None
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core.schema import Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
    # Create a dummy Document class for type hints when llama-index is not available
    class Document:
        pass
    logging.warning("llama-index not available. Install with: pip install llama-index")

class RAGEngine:
    """RAG engine for semantic search over transcripts"""
    
    def __init__(self, config: Dict[str, Any], transcripts_dir: Optional[str] = None):
        self.config = config
        self.rag_config = config['rag']
        self.logger = logging.getLogger("parenting_ai.rag")
        
        if not HAS_LLAMA_INDEX:
            raise ImportError("llama-index is required but not installed")
        
        # Paths - use specified directory or check both local and parent directory for transcripts
        self.transcripts_dirs = []
        
        if transcripts_dir:
            # Use specified directory
            specified_dir = Path(transcripts_dir)
            if specified_dir.exists():
                self.transcripts_dirs.append(specified_dir)
                self.logger.info(f"Using specified transcripts directory: {specified_dir}")
            else:
                self.logger.warning(f"Specified transcripts directory does not exist: {specified_dir}")
        else:
            # Default behavior: check both local and parent directory
            local_transcripts = Path("transcriptions")
            parent_transcripts = Path("../transcriptions")
            
            if local_transcripts.exists():
                self.transcripts_dirs.append(local_transcripts)
                self.logger.info(f"Found transcripts directory: {local_transcripts}")
            
            if parent_transcripts.exists():
                self.transcripts_dirs.append(parent_transcripts)
                self.logger.info(f"Found transcripts directory: {parent_transcripts}")
        
        if not self.transcripts_dirs:
            self.logger.warning("No transcripts directory found in 'transcriptions' or '../transcriptions'")
            # Set default for compatibility
            self.transcripts_dirs = [Path(transcripts_dir) if transcripts_dir else Path("transcriptions")]
        
        self.index_dir = Path("parenting_ai_index")
        try:
            self.index_dir.mkdir(exist_ok=True)
            # Check write permissions
            test_file = self.index_dir / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError) as e:
                self.logger.warning(f"Index directory {self.index_dir} is not writable: {e}. Index will not be persisted.")
        except (PermissionError, OSError) as e:
            self.logger.error(f"Failed to create index directory {self.index_dir}: {e}")
            raise
        
        # Initialize components
        self.embedding_model = None
        self.vector_store = None
        self.index = None
        self.query_engine = None
        self.node_parser = None
        
        # Initialize embedding model and node parser from config
        self._initialize_embedding_model()
        self._initialize_node_parser()
        
        # Load or create index
        self._initialize_index()
    
    def _initialize_embedding_model(self) -> None:
        """Initialize embedding model from config"""
        try:
            use_mock = self.rag_config.get('use_mock_embeddings', False)
            
            if use_mock:
                # Use mock embeddings for debugging/testing
                self.logger.info("Using MockEmbedding (debug mode)")
                self.embedding_model = MockEmbedding(embed_dim=384)
            else:
                # Use actual embedding model from config
                if not HAS_HUGGINGFACE or HuggingFaceEmbedding is None:
                    self.logger.warning("HuggingFaceEmbedding not available. Falling back to MockEmbedding.")
                    self.embedding_model = MockEmbedding(embed_dim=384)
                else:
                    embedding_model_name = self.rag_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                    self.logger.info(f"Loading embedding model: {embedding_model_name}")
                    try:
                        self.embedding_model = HuggingFaceEmbedding(model_name=embedding_model_name)
                        self.logger.info(f"Embedding model loaded: {embedding_model_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load embedding model {embedding_model_name}: {e}")
                        raise
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize embedding model: {e}. Falling back to MockEmbedding.")
            self.embedding_model = MockEmbedding(embed_dim=384)
    
    def _initialize_node_parser(self) -> None:
        """Initialize node parser with chunk size from config"""
        chunk_size = self.rag_config.get('chunk_size', 512)
        self.node_parser = SentenceSplitter(chunk_size=chunk_size)
        self.logger.info(f"Initialized node parser with chunk_size={chunk_size}")
    
    def _initialize_index(self) -> None:
        """Initialize or load the vector index"""
        try:
            # Try to load existing index
            if self._load_existing_index():
                self.logger.info("Loaded existing vector index")
                return
            
            # Create new index
            self.logger.info("Creating new vector index...")
            self._create_new_index()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize index: {e}")
            raise
    
    def _load_existing_index(self) -> bool:
        """Try to load existing index from storage"""
        try:
            storage_path = self.index_dir / "storage"
            if not storage_path.exists():
                return False
            
            # Ensure embedding model is initialized (needed for loading index)
            if self.embedding_model is None:
                self._initialize_embedding_model()
            
            # Load storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=str(storage_path)
            )
            
            # Load index (will use the embedding model that matches what was used during indexing)
            self.index = load_index_from_storage(storage_context, embed_model=self.embedding_model)
            
            # Create query engine with Ollama LLM
            from llama_index.llms.ollama import Ollama
            llm = Ollama(model=self.config['llm']['model'], request_timeout=self.config['llm']['timeout'])
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.rag_config.get('top_k', 5),
                llm=llm
            )
            
            self.logger.info(f"Loaded index with {len(self.index.docstore.docs)} documents")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing index: {e}")
            return False
    
    def _create_new_index(self) -> None:
        """Create new vector index from transcripts"""
        if self.embedding_model is None:
            self._initialize_embedding_model()
        
        # Get embedding dimension from model
        try:
            # Try to get embed_dim from the model
            if hasattr(self.embedding_model, 'dim'):
                embed_dim = self.embedding_model.dim
            elif hasattr(self.embedding_model, 'embed_dim'):
                embed_dim = self.embedding_model.embed_dim
            else:
                # Default dimension for sentence-transformers/all-MiniLM-L6-v2
                embed_dim = 384
        except:
            # Fallback dimension
            embed_dim = 384
        
        # Create vector store with appropriate dimension
        import faiss
        self.vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(embed_dim))
        
        # Load and process documents
        documents = self._load_transcript_documents()
        
        # Process documents with node parser if available
        if self.node_parser and documents:
            self.logger.info(f"Processing {len(documents)} documents with chunk_size={self.rag_config.get('chunk_size', 512)}")
        
        if not documents:
            self.logger.warning("No transcript documents found")
            # Create empty index
            self.index = VectorStoreIndex.from_documents(
                [], 
                embed_model=self.embedding_model,
                vector_store=self.vector_store,
                node_parser=self.node_parser
            )
        else:
            # Create index from documents
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embedding_model,
                vector_store=self.vector_store,
                node_parser=self.node_parser
            )
            
            # Persist index
            self._persist_index()
        
        # Create query engine with Ollama LLM
        from llama_index.llms.ollama import Ollama
        llm = Ollama(model=self.config['llm']['model'], request_timeout=self.config['llm']['timeout'])
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.rag_config['top_k'],
            llm=llm
        )
        
        self.logger.info(f"Created index with {len(documents)} documents using embedding_model={self.rag_config.get('embedding_model', 'N/A')}")
    
    def _load_transcript_documents(self) -> List[Document]:
        """Load and process transcript JSON files from multiple possible locations"""
        documents = []
        all_json_files = []
        
        # Search all transcript directories
        for transcripts_dir in self.transcripts_dirs:
            if transcripts_dir.exists():
                json_files = list(transcripts_dir.rglob("*.json"))
                self.logger.info(f"Found {len(json_files)} JSON files in {transcripts_dir}")
                all_json_files.extend(json_files)
            else:
                self.logger.debug(f"Transcripts directory does not exist: {transcripts_dir}")
        
        if not all_json_files:
            self.logger.warning(f"No transcript JSON files found in any transcripts directory")
            return documents
        
        # Limit to 10 files for initial testing (can be removed later for production)
        json_files_to_process = all_json_files[:10]
        self.logger.info(f"Processing {len(json_files_to_process)} transcript files (limited for testing)")
        
        for json_file in json_files_to_process:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract transcript data
                transcript_docs = self._process_transcript_file(json_file, data)
                documents.extend(transcript_docs)
                
            except Exception as e:
                self.logger.warning(f"Failed to process {json_file}: {e}")
                continue
        
        self.logger.info(f"Loaded {len(documents)} document chunks from {len(json_files_to_process)} transcript files")
        return documents
    
    def _process_transcript_file(self, file_path: Path, data: Dict[str, Any]) -> List[Document]:
        """Process a single transcript file into Document objects"""
        documents = []
        
        # Extract basic info
        timestamp = data.get('timestamp', '')
        audio_file = data.get('audio_file', '')
        model = data.get('model', '')
        
        # Extract location from filename
        location = self._extract_location_from_path(file_path)
        
        # Process segments
        segments = data.get('transcription', {}).get('segments', [])
        merged_segments = data.get('merged_segments', [])
        
        # Use merged segments if available, otherwise use regular segments
        segments_to_process = merged_segments if merged_segments else segments
        
        for segment in segments_to_process:
            # Create document for each segment
            text = segment.get('text', '').strip()
            if not text:
                continue
            
            # Extract metadata
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            speaker = segment.get('speaker', 'Unknown')
            
            # Create document
            doc = Document(
                text=text,
                metadata={
                    'timestamp': timestamp,
                    'audio_file': audio_file,
                    'location': location,
                    'speaker': speaker,
                    'start_time': start_time,
                    'end_time': end_time,
                    'model': model,
                    'file_path': str(file_path)
                }
            )
            documents.append(doc)
        
        return documents
    
    def _extract_location_from_path(self, file_path: Path) -> str:
        """Extract location from file path"""
        filename = file_path.name
        
        # Pattern: recording_YYYYMMDD_HHMMSS_Location Name.wav
        import re
        match = re.search(r'recording_\d{8}_\d{6}_(.+?)(?:_partial)?\.wav', filename)
        if match:
            return match.group(1)
        
        # Pattern: transcript_recording_YYYYMMDD_HHMMSS_chN
        match = re.search(r'transcript_recording_\d{8}_\d{6}_ch(\d+)', filename)
        if match:
            channel = int(match.group(1))
            locations = {
                1: "TV Living Room",
                2: "Dining Room", 
                3: "Rowe Bedroom",
                4: "Penn Bedroom"
            }
            return locations.get(channel, f"Channel {channel}")
        
        return "Unknown Location"
    
    def _persist_index(self) -> None:
        """Persist the index to disk"""
        try:
            storage_path = self.index_dir / "storage"
            # Ensure storage directory exists and is writable
            storage_path.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(storage_path))
            self.logger.info(f"Index persisted to {storage_path}")
        except (PermissionError, OSError) as e:
            self.logger.error(f"Failed to persist index due to permission error: {e}")
            self.logger.error(f"Please fix permissions: sudo chown -R {os.getenv('USER', 'user')}:{os.getenv('USER', 'user')} {self.index_dir}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to persist index: {e}")
            raise
    
    def search(self, query: str, time_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant transcript segments"""
        try:
            if not self.query_engine:
                self.logger.error("Query engine not initialized")
                return []
            
            # Perform search
            response = self.query_engine.query(query)
            
            # Extract results
            results = []
            for node in response.source_nodes:
                result = {
                    'text': node.text,
                    'score': node.score,
                    'metadata': node.metadata,
                    'timestamp': node.metadata.get('timestamp', ''),
                    'speaker': node.metadata.get('speaker', 'Unknown'),
                    'location': node.metadata.get('location', 'Unknown'),
                    'start_time': node.metadata.get('start_time', 0),
                    'end_time': node.metadata.get('end_time', 0)
                }
                
                # Apply time filter if provided
                if time_filter and not self._matches_time_filter(result, time_filter):
                    continue
                
                results.append(result)
            
            # Apply similarity threshold filter if configured
            similarity_threshold = self.rag_config.get('similarity_threshold', 0.0)
            if similarity_threshold > 0:
                # Note: scores in llama-index are typically distances (lower is better)
                # For cosine similarity (higher is better), we may need to convert
                # Filter out results below threshold
                filtered_results = []
                for result in results:
                    # Check if result meets similarity threshold
                    # Assuming score is similarity (higher = better)
                    if result.get('score', 0) >= similarity_threshold:
                        filtered_results.append(result)
                results = filtered_results
                self.logger.info(f"Filtered to {len(results)} results above similarity threshold {similarity_threshold}")
            
            # Sort by score (highest first)
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            self.logger.info(f"Found {len(results)} relevant segments for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _matches_time_filter(self, result: Dict[str, Any], time_filter: Dict[str, Any]) -> bool:
        """Check if result matches time filter"""
        if not time_filter or 'start_time' not in time_filter:
            return True
        
        try:
            result_timestamp = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
            filter_start = time_filter['start_time']
            
            return result_timestamp >= filter_start
        except Exception:
            return True
    
    def rebuild_index(self) -> None:
        """Rebuild the entire index from scratch"""
        self.logger.info("Rebuilding index...")
        
        # Remove existing index
        import shutil
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Create new index
        self._create_new_index()
        self.logger.info("Index rebuild complete")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        try:
            if not self.index:
                return {'error': 'Index not initialized'}
            
            return {
                'document_count': len(self.index.docstore.docs),
                'index_dir': str(self.index_dir),
                'embedding_model': self.rag_config.get('embedding_model', 'N/A'),
                'use_mock_embeddings': self.rag_config.get('use_mock_embeddings', False),
                'chunk_size': self.rag_config.get('chunk_size', 512),
                'top_k': self.rag_config.get('top_k', 5),
                'similarity_threshold': self.rag_config.get('similarity_threshold', 0.7),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def test_search(self, query: str = "bedtime routine") -> Dict[str, Any]:
        """Test the search functionality"""
        try:
            results = self.search(query)
            
            return {
                'status': 'success',
                'query': query,
                'result_count': len(results),
                'top_result': results[0] if results else None,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }



