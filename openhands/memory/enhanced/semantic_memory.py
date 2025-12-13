"""Semantic code memory for long-term codebase understanding."""

from __future__ import annotations

import ast
import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import networkx as nx
from sentence_transformers import SentenceTransformer

from openhands.core.logger import openhands_logger as logger

from .types import CodeChunk, Symbol


class SemanticCodeMemory:
    """Manages semantic understanding of codebases using vector embeddings and graph relationships."""

    def __init__(self, repo_path: str, memory_dir: str = '.openhands/memory'):
        self.repo_path = Path(repo_path)
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.memory_dir / 'chroma_db')
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name='code_chunks', metadata={'hnsw:space': 'cosine'}
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize dependency graph
        self.dependency_graph = nx.DiGraph()

        # Initialize SQLite for metadata
        self.db_path = self.memory_dir / 'semantic_memory.db'
        self._init_database()

        # File tracking
        self.indexed_files: dict[str, str] = {}  # file_path -> content_hash
        self._load_indexed_files()

    def _init_database(self) -> None:
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_chunks (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    chunk_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    symbols TEXT,  -- JSON array of symbols
                    dependencies TEXT,  -- JSON array of dependencies
                    complexity INTEGER DEFAULT 1,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    scope TEXT,
                    signature TEXT,
                    docstring TEXT,
                    chunk_id TEXT,
                    FOREIGN KEY (chunk_id) REFERENCES code_chunks (id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_index (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0
                )
            """)

            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols (name)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols (file_path)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_chunks_file ON code_chunks (file_path)'
            )

    def index_repository(self, force_reindex: bool = False) -> None:
        """Index the entire repository."""
        logger.info(f'Indexing repository: {self.repo_path}')

        for file_path in self._get_code_files():
            try:
                self.index_file(file_path, force_reindex)
            except Exception as e:
                logger.warning(f'Failed to index {file_path}: {e}')

        logger.info('Repository indexing complete')

    def index_file(self, file_path: str, force_reindex: bool = False) -> None:
        """Index a single file."""
        full_path = self.repo_path / file_path
        if not full_path.exists():
            return

        # Check if file needs reindexing
        content = full_path.read_text(encoding='utf-8', errors='ignore')
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if not force_reindex and file_path in self.indexed_files:
            if self.indexed_files[file_path] == content_hash:
                return  # File unchanged

        logger.debug(f'Indexing file: {file_path}')

        # Remove old chunks for this file
        self._remove_file_chunks(file_path)

        # Parse and chunk the file
        chunks = self._chunk_file(file_path, content)

        # Store chunks
        for chunk in chunks:
            self._store_chunk(chunk)

        # Update file index
        self.indexed_files[file_path] = content_hash
        self._update_file_index(file_path, content_hash, len(chunks))

    def retrieve_relevant_code(
        self, query: str, file_patterns: list[str] | None = None, max_results: int = 10
    ) -> list[CodeChunk]:
        """Retrieve code chunks relevant to the query."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max_results * 2,  # Get more results for filtering
            include=['metadatas', 'documents', 'distances'],
        )

        chunks = []
        for i, (metadata, document, distance) in enumerate(
            zip(
                results['metadatas'][0],
                results['documents'][0],
                results['distances'][0],
            )
        ):
            # Filter by file patterns if provided
            if file_patterns:
                file_path = metadata['file_path']
                if not any(pattern in file_path for pattern in file_patterns):
                    continue

            # Convert back to CodeChunk
            chunk = self._metadata_to_chunk(metadata, document)
            chunk.metadata['relevance_score'] = (
                1.0 - distance
            )  # Convert distance to similarity
            chunks.append(chunk)

            if len(chunks) >= max_results:
                break

        return chunks

    def get_file_dependencies(self, file_path: str) -> list[str]:
        """Get dependencies for a file."""
        if file_path in self.dependency_graph:
            return list(self.dependency_graph.successors(file_path))
        return []

    def get_file_dependents(self, file_path: str) -> list[str]:
        """Get files that depend on this file."""
        if file_path in self.dependency_graph:
            return list(self.dependency_graph.predecessors(file_path))
        return []

    def find_symbols(
        self, symbol_name: str, symbol_type: str | None = None
    ) -> list[Symbol]:
        """Find symbols by name and optionally type."""
        with sqlite3.connect(self.db_path) as conn:
            query = 'SELECT * FROM symbols WHERE name = ?'
            params = [symbol_name]

            if symbol_type:
                query += ' AND symbol_type = ?'
                params.append(symbol_type)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            symbols = []
            for row in rows:
                symbols.append(
                    Symbol(
                        name=row[1],
                        symbol_type=row[2],
                        file_path=row[3],
                        line_number=row[4],
                        scope=row[5] or '',
                        signature=row[6],
                        docstring=row[7],
                    )
                )

            return symbols

    def get_file_symbols(self, file_path: str) -> list[Symbol]:
        """Get all symbols defined in a file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT * FROM symbols WHERE file_path = ?', [file_path]
            )
            rows = cursor.fetchall()

            symbols = []
            for row in rows:
                symbols.append(
                    Symbol(
                        name=row[1],
                        symbol_type=row[2],
                        file_path=row[3],
                        line_number=row[4],
                        scope=row[5] or '',
                        signature=row[6],
                        docstring=row[7],
                    )
                )

            return symbols

    def _get_code_files(self) -> list[str]:
        """Get list of code files to index."""
        code_extensions = {
            '.py',
            '.js',
            '.ts',
            '.java',
            '.cpp',
            '.c',
            '.h',
            '.go',
            '.rs',
            '.rb',
        }
        code_files = []

        for root, dirs, files in os.walk(self.repo_path):
            # Skip common non-code directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith('.')
                and d
                not in {'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist'}
            ]

            for file in files:
                if Path(file).suffix in code_extensions:
                    rel_path = os.path.relpath(os.path.join(root, file), self.repo_path)
                    code_files.append(rel_path)

        return code_files

    def _chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Chunk a file into semantic units."""
        chunks = []

        if file_path.endswith('.py'):
            chunks.extend(self._chunk_python_file(file_path, content))
        else:
            # Fallback: chunk by lines
            chunks.extend(self._chunk_by_lines(file_path, content))

        return chunks

    def _chunk_python_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Chunk a Python file by AST nodes."""
        chunks = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to line-based chunking
            return self._chunk_by_lines(file_path, content)

        lines = content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno
                end_line = node.end_lineno or start_line

                chunk_content = '\n'.join(lines[start_line - 1 : end_line])
                chunk_type = (
                    'function'
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else 'class'
                )

                # Extract symbols
                symbols = [node.name]
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            symbols.append(f'{node.name}.{item.name}')

                # Calculate complexity (simplified)
                complexity = self._calculate_complexity(node)

                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    symbols=symbols,
                    complexity=complexity,
                    last_modified=datetime.now(),
                )

                chunks.append(chunk)

        return chunks

    def _chunk_by_lines(
        self, file_path: str, content: str, chunk_size: int = 50
    ) -> list[CodeChunk]:
        """Fallback chunking by lines."""
        lines = content.split('\n')
        chunks = []

        for i in range(0, len(lines), chunk_size):
            start_line = i + 1
            end_line = min(i + chunk_size, len(lines))
            chunk_content = '\n'.join(lines[i:end_line])

            chunk = CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type='block',
                complexity=1,
                last_modified=datetime.now(),
            )

            chunks.append(chunk)

        return chunks

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def _store_chunk(self, chunk: CodeChunk) -> None:
        """Store a code chunk in the vector database and metadata store."""
        # Generate embedding
        embedding = self.embedding_model.encode([chunk.content])[0]
        chunk.embedding = embedding

        # Generate unique ID
        chunk_id = hashlib.md5(
            f'{chunk.file_path}:{chunk.start_line}:{chunk.end_line}'.encode()
        ).hexdigest()

        # Store in vector database
        self.collection.add(
            ids=[chunk_id],
            embeddings=[embedding.tolist()],
            documents=[chunk.content],
            metadatas=[
                {
                    'file_path': chunk.file_path,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'chunk_type': chunk.chunk_type,
                    'symbols': ','.join(chunk.symbols),
                    'complexity': chunk.complexity,
                }
            ],
        )

        # Store metadata in SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO code_chunks
                (id, file_path, start_line, end_line, chunk_type, content_hash,
                 symbols, dependencies, complexity, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    chunk_id,
                    chunk.file_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.chunk_type,
                    hashlib.md5(chunk.content.encode()).hexdigest(),
                    ','.join(chunk.symbols),
                    ','.join(chunk.dependencies),
                    chunk.complexity,
                    chunk.last_modified,
                ),
            )

    def _remove_file_chunks(self, file_path: str) -> None:
        """Remove all chunks for a file."""
        with sqlite3.connect(self.db_path) as conn:
            # Get chunk IDs to remove from vector database
            cursor = conn.execute(
                'SELECT id FROM code_chunks WHERE file_path = ?', [file_path]
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]

            # Remove from vector database
            if chunk_ids:
                try:
                    self.collection.delete(ids=chunk_ids)
                except Exception as e:
                    logger.warning(f'Failed to remove chunks from vector DB: {e}')

            # Remove from SQLite
            conn.execute('DELETE FROM code_chunks WHERE file_path = ?', [file_path])
            conn.execute('DELETE FROM symbols WHERE file_path = ?', [file_path])

    def _metadata_to_chunk(self, metadata: dict[str, Any], content: str) -> CodeChunk:
        """Convert metadata back to CodeChunk."""
        return CodeChunk(
            content=content,
            file_path=metadata['file_path'],
            start_line=metadata['start_line'],
            end_line=metadata['end_line'],
            chunk_type=metadata['chunk_type'],
            symbols=metadata.get('symbols', '').split(',')
            if metadata.get('symbols')
            else [],
            complexity=metadata.get('complexity', 1),
        )

    def _load_indexed_files(self) -> None:
        """Load the index of already processed files."""
        if not self.db_path.exists():
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT file_path, content_hash FROM file_index')
            for file_path, content_hash in cursor.fetchall():
                self.indexed_files[file_path] = content_hash

    def _update_file_index(
        self, file_path: str, content_hash: str, chunk_count: int
    ) -> None:
        """Update the file index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_index
                (file_path, content_hash, last_indexed, chunk_count)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            """,
                (file_path, content_hash, chunk_count),
            )
