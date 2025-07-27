"""
Chunking strategies for text processing using LangChain.

This module provides various text chunking strategies that can be used for
processing documents in RAG (Retrieval-Augmented Generation) pipelines.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain.schema import Document
import tiktoken
import pandas as pd
from bs4 import BeautifulSoup
import re
from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import json
import io
from pathlib import Path

@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any] = None
    chunk_id: str = ""

class ChunkingLogic:
    """
    A class that provides various text chunking strategies.
    
    This class implements multiple chunking strategies using LangChain's text splitters
    and provides a unified interface for chunking text.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the ChunkingLogic with default chunk size and overlap.
        
        Args:
            chunk_size: The target size of each chunk (in characters or tokens)
            chunk_overlap: The amount of overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def recursive_character_chunking(
        self, 
        text: str, 
        separators: List[str] = None,
        **kwargs
    ) -> List[Chunk]:
        """
        Split text recursively by different characters to find the best split.
        
        Args:
            text: The text to split
            separators: List of separators to use for splitting
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List of Chunk objects
        """
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            **kwargs
        )
        
        chunks = splitter.split_text(text)
        return [Chunk(text=chunk) for chunk in chunks]
    
    def character_chunking(
        self, 
        text: str, 
        separator: str = "\n\n",
        **kwargs
    ) -> List[Chunk]:
        """
        Split text by character count with overlap.
        
        Args:
            text: The text to split
            separator: The separator to use for splitting
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List of Chunk objects
        """
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=separator,
            **kwargs
        )
        
        chunks = splitter.split_text(text)
        return [Chunk(text=chunk) for chunk in chunks]
    
    def token_chunking(
        self, 
        text: str, 
        encoding_name: str = "cl100k_base",
        **kwargs
    ) -> List[Chunk]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: The text to split
            encoding_name: The tokenizer encoding to use
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List of Chunk objects
        """
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=encoding_name,
            **kwargs
        )
        
        chunks = splitter.split_text(text)
        return [Chunk(text=chunk) for chunk in chunks]
    
    def markdown_chunking(
        self, 
        text: str,
        **kwargs
    ) -> List[Chunk]:
        """
        Split markdown text based on markdown headers and sections.
        
        Args:
            text: The markdown text to split
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List of Chunk objects
        """
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **kwargs
        )
        
        chunks = splitter.split_text(text)
        return [Chunk(text=chunk) for chunk in chunks]
    
    def python_code_chunking(
        self, 
        text: str,
        **kwargs
    ) -> List[Chunk]:
        """
        Split Python code while preserving function and class boundaries.
        
        Args:
            text: The Python code to split
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List of Chunk objects
        """
        splitter = PythonCodeTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **kwargs
        )
        
        chunks = splitter.split_text(text)
        return [Chunk(text=chunk) for chunk in chunks]
    
    def nltk_sentence_chunking(
        self, 
        text: str,
        **kwargs
    ) -> List[Chunk]:
        """
        Split text into chunks based on NLTK sentence tokenization.
        
        Args:
            text: The text to split
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List of Chunk objects
        """
        splitter = NLTKTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **kwargs
        )
        
        chunks = splitter.split_text(text)
        return [Chunk(text=chunk) for chunk in chunks]
    
    def spacy_chunking(
        self, 
        text: str,
        model_name: str = "en_core_web_sm",
        **kwargs
    ) -> List[Chunk]:
        """
        Split text into chunks using spaCy's sentence segmentation.
        
        Args:
            text: The text to split
            model_name: The spaCy model to use
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List of Chunk objects
        """
        splitter = SpacyTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            pipeline=model_name,
            **kwargs
        )
        
        chunks = splitter.split_text(text)
        return [Chunk(text=chunk) for chunk in chunks]
    
    def sentence_transformers_chunking(
        self, 
        text: str,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        **kwargs
    ) -> List[Chunk]:
        """
        Split text into chunks using a Sentence Transformers tokenizer.
        
        Args:
            text: The text to split
            model_name: The Sentence Transformers model to use
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List of Chunk objects
        """
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model_name=model_name,
            **kwargs
        )
        
        chunks = splitter.split_text(text)
        return [Chunk(text=chunk) for chunk in chunks]
    
    def semantic_chunking(
        self,
        text: str,
        threshold: float = 0.8,
        **kwargs
    ) -> List[Chunk]:
        """
        Split text into semantically meaningful chunks.
        
        This is a more advanced chunking strategy that tries to maintain
        semantic coherence within each chunk.
        
        Args:
            text: The text to split
            threshold: The similarity threshold for combining chunks
            **kwargs: Additional arguments
            
        Returns:
            List of Chunk objects
        """
        # First, split by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and current_length + para_length > self.chunk_size:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(Chunk(text=chunk_text))
                
                # Keep some context for the next chunk
                overlap = self.chunk_overlap // 100
                current_chunk = current_chunk[-overlap:] if overlap else []
                current_length = sum(len(p) for p in current_chunk)
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(Chunk(text=chunk_text))
        
        return chunks
        
    def table_chunking(
        self,
        text: str,
        table_processor: Optional[Callable] = None,
        **kwargs
    ) -> List[Chunk]:
        """
        Extract and process tables from text.
        
        Args:
            text: Text potentially containing tables
            table_processor: Function to process table content
            **kwargs: Additional arguments
            
        Returns:
            List of Chunk objects containing tables and surrounding text
        """
        # Simple table detection pattern (can be enhanced)
        table_pattern = r'\|(.+\|)+\s*\n\|[-|\s]+\n((?:\|.*\|\s*\n)+)'
        
        chunks = []
        last_end = 0
        
        for match in re.finditer(table_pattern, text, re.MULTILINE):
            # Add text before table
            if match.start() > last_end:
                chunks.append(Chunk(
                    text=text[last_end:match.start()].strip(),
                    metadata={"content_type": "text"}
                ))
            
            # Process table
            table_text = match.group(0)
            if table_processor:
                try:
                    processed = table_processor(table_text)
                    table_text = str(processed)
                except Exception as e:
                    print(f"Error processing table: {e}")
            
            chunks.append(Chunk(
                text=table_text,
                metadata={"content_type": "table"}
            ))
            
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            chunks.append(Chunk(
                text=text[last_end:].strip(),
                metadata={"content_type": "text"}
            ))
            
        return chunks
    
    def html_chunking(
        self,
        html_content: str,
        preserve_structure: bool = True,
        **kwargs
    ) -> List[Chunk]:
        """
        Chunk HTML content while preserving document structure.
        
        Args:
            html_content: HTML content to chunk
            preserve_structure: Whether to preserve HTML structure in chunks
            **kwargs: Additional arguments
            
        Returns:
            List of Chunk objects
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if not preserve_structure:
            # Simple text extraction
            text = soup.get_text('\n', strip=True)
            return self.recursive_character_chunking(text, **kwargs)
        
        # Preserve structure by chunking by semantic elements
        chunks = []
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table', 'ul', 'ol']):
            element_text = element.get_text('\n', strip=True)
            if element_text:
                chunks.append(Chunk(
                    text=element_text,
                    metadata={
                        "tag": element.name,
                        "class": element.get('class', []),
                        "id": element.get('id')
                    }
                ))
                
        return chunks
    
    def latex_chunking(
        self,
        latex_content: str,
        preserve_structure: bool = True,
        **kwargs
    ) -> List[Chunk]:
        """
        Chunk LaTeX content while preserving document structure.
        
        Args:
            latex_content: LaTeX content to chunk
            preserve_structure: Whether to preserve LaTeX structure
            **kwargs: Additional arguments
            
        Returns:
            List of Chunk objects
        """
        if not preserve_structure:
            # Simple text extraction (remove LaTeX commands)
            text = re.sub(r'\\(?:[a-zA-Z]+|.)', ' ', latex_content)
            return self.recursive_character_chunking(text, **kwargs)
            
        # More sophisticated LaTeX parsing
        chunks = []
        
        # Split by sections
        sections = re.split(r'(\\(?:sub)*section\*?\{.*?\})', latex_content)
        
        current_chunk = []
        current_length = 0
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            if section.startswith('\\'):
                # This is a section header
                if current_chunk:
                    chunks.append(Chunk(
                        text=''.join(current_chunk),
                        metadata={"content_type": "section"}
                    ))
                    current_chunk = []
                    current_length = 0
                
                # Add section as its own chunk
                chunks.append(Chunk(
                    text=section.strip(),
                    metadata={"content_type": "section_header"}
                ))
            else:
                # Regular content
                section_length = len(section)
                if current_length + section_length > self.chunk_size and current_chunk:
                    chunks.append(Chunk(
                        text=''.join(current_chunk),
                        metadata={"content_type": "text"}
                    ))
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(section)
                current_length += section_length
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(Chunk(
                text=''.join(current_chunk),
                metadata={"content_type": "text"}
            ))
            
        return chunks
    
    def hybrid_chunking(
        self,
        text: str,
        strategies: List[Tuple[str, dict]] = None,
        **kwargs
    ) -> List[Chunk]:
        """
        Apply multiple chunking strategies in sequence.
        
        Args:
            text: Text to chunk
            strategies: List of (strategy_name, params) tuples
            **kwargs: Default parameters for all strategies
            
        Returns:
            List of Chunk objects
        """
        if not strategies:
            # Default strategy sequence
            strategies = [
                ("table", {}),
                ("recursive", {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap})
            ]
        
        current_text = text
        chunks = []
        
        for strategy_name, params in strategies:
            if not current_text:
                break
                
            # Get the strategy function
            strategy_fn = get_chunking_strategy(strategy_name, **{**kwargs, **params})
            
            # Apply the strategy
            new_chunks = strategy_fn(current_text)
            
            # If this is the first strategy, use its chunks directly
            if not chunks:
                chunks = new_chunks
            else:
                # Otherwise, try to merge chunks intelligently
                # (This is a simple implementation - could be enhanced)
                merged_chunks = []
                for chunk in new_chunks:
                    if not merged_chunks:
                        merged_chunks.append(chunk)
                    else:
                        last_chunk = merged_chunks[-1]
                        if len(last_chunk.text) + len(chunk.text) <= self.chunk_size:
                            # Merge chunks if they're small
                            last_chunk.text += "\n\n" + chunk.text
                            # Merge metadata (simple implementation)
                            if hasattr(last_chunk, 'metadata') and hasattr(chunk, 'metadata'):
                                last_chunk.metadata.update(chunk.metadata)
                            elif hasattr(chunk, 'metadata'):
                                last_chunk.metadata = chunk.metadata
                        else:
                            merged_chunks.append(chunk)
                
                chunks = merged_chunks
            
            # Prepare for next strategy (if any)
            if len(strategies) > 1:
                current_text = "\n\n".join(chunk.text for chunk in chunks)
                chunks = []
        
        return chunks
        
    def evaluate_chunks(self, chunks: List[Chunk]) -> Dict[str, float]:
        """
        Evaluate the quality of chunks based on various metrics.
        
        Args:
            chunks: List of Chunk objects to evaluate
            
        Returns:
            Dictionary containing evaluation metrics:
            - chunk_count: Total number of chunks
            - avg_chunk_size: Average character count per chunk
            - min_chunk_size: Smallest chunk size in characters
            - max_chunk_size: Largest chunk size in characters
            - size_std_dev: Standard deviation of chunk sizes
            - avg_token_count: Average token count per chunk (approximate)
            - content_types: Distribution of content types (if available)
            - avg_sentence_count: Average number of sentences per chunk
            - avg_word_count: Average number of words per chunk
            - compression_ratio: Ratio of original text to chunked text size
            - overlap_ratio: Estimated overlap between chunks (if metadata available)
        """
        if not chunks:
            return {}
            
        import numpy as np
        from collections import defaultdict
        import re
        
        # Initialize metrics
        chunk_sizes = []
        token_counts = []
        sentence_counts = []
        word_counts = []
        content_types = defaultdict(int)
        original_text = ""
        
        # Simple tokenizer (approximate)
        def count_tokens(text: str) -> int:
            # Rough estimate: 1 token ≈ 4 characters or 0.75 words (English)
            return max(len(text) // 4, len(text.split()))
            
        # Simple sentence counter (approximate)
        def count_sentences(text: str) -> int:
            # Count sentence-ending punctuation
            return len(re.findall(r'[.!?]+\s+', text + ' '))
        
        # Process each chunk
        for chunk in chunks:
            text = chunk.text
            chunk_sizes.append(len(text))
            token_count = count_tokens(text)
            token_counts.append(token_count)
            sentence_count = count_sentences(text)
            sentence_counts.append(sentence_count)
            word_count = len(text.split())
            word_counts.append(word_count)
            
            # Track content types if available in metadata
            if hasattr(chunk, 'metadata') and chunk.metadata:
                content_type = chunk.metadata.get('content_type', 'unknown')
                content_types[content_type] += 1
            
            # Build original text (for compression ratio)
            original_text += text
        
        # Calculate metrics
        metrics = {
            'chunk_count': len(chunks),
            'avg_chunk_size': float(np.mean(chunk_sizes)),
            'min_chunk_size': float(np.min(chunk_sizes)),
            'max_chunk_size': float(np.max(chunk_sizes)),
            'size_std_dev': float(np.std(chunk_sizes)),
            'avg_token_count': float(np.mean(token_counts)),
            'avg_sentence_count': float(np.mean(sentence_counts) if sentence_counts else 0),
            'avg_word_count': float(np.mean(word_counts)),
            'compression_ratio': len(original_text) / sum(chunk_sizes) if chunk_sizes else 1.0,
        }
        
        # Add content type distribution if available
        if content_types:
            metrics['content_types'] = dict(content_types)
            
        # Calculate overlap ratio if metadata contains overlap information
        if hasattr(chunks[0], 'metadata') and 'overlap' in chunks[0].metadata:
            overlaps = [c.metadata.get('overlap', 0) for c in chunks if hasattr(c, 'metadata')]
            if overlaps:
                metrics['avg_overlap_ratio'] = float(np.mean(overlaps))
        
        return metrics

def get_chunking_strategy(strategy_name: str, **kwargs) -> callable:
    """
    Get a chunking strategy function by name.
    
    Args:
        strategy_name: Name of the chunking strategy
        **kwargs: Arguments to pass to the chunking strategy
        
    Returns:
        A callable chunking function
        
    Raises:
        ValueError: If the strategy name is not recognized
    """
    chunker = ChunkingLogic(
        chunk_size=kwargs.pop('chunk_size', 1000),
        chunk_overlap=kwargs.pop('chunk_overlap', 200)
    )
    
    strategies = {
        # Basic strategies
        "recursive": chunker.recursive_character_chunking,
        "character": chunker.character_chunking,
        "token": chunker.token_chunking,
        "semantic": chunker.semantic_chunking,
        
        # Document-specific strategies
        "markdown": chunker.markdown_chunking,
        "python": chunker.python_code_chunking,
        "table": chunker.table_chunking,
        "html": chunker.html_chunking,
        "latex": chunker.latex_chunking,
        
        # Advanced strategies
        "nltk": chunker.nltk_sentence_chunking,
        "spacy": chunker.spacy_chunking,
        "sentence_transformers": chunker.sentence_transformers_chunking,
        "hybrid": chunker.hybrid_chunking,
    }
    
    if strategy_name not in strategies:
        available = ", ".join(f'"{s}"' for s in strategies.keys())
        raise ValueError(
            f"Unknown chunking strategy: '{strategy_name}'. "
            f"Available strategies are: {available}"
        )
    
    # Return a function that calls the strategy with the remaining kwargs
    def strategy_wrapper(text: str):
        return strategies[strategy_name](text, **kwargs)
        
    return strategy_wrapper