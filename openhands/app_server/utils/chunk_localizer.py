"""Chunk localizer to help localize the most relevant chunks in a file.

This is primarily used to localize the most relevant chunks in a file
for a given query (e.g. edit draft produced by the agent).
"""

from pydantic import BaseModel
from rapidfuzz.distance import LCSseq
from tree_sitter import Node, Tree
from tree_sitter_language_pack import get_parser

from openhands.app_server.utils.logger import openhands_logger as logger


class Chunk(BaseModel):
    text: str
    line_range: tuple[int, int]  # (start_line, end_line), 1-index, inclusive
    normalized_lcs: float | None = None

    def visualize(self) -> str:
        lines = self.text.split('\n')
        assert len(lines) == self.line_range[1] - self.line_range[0] + 1
        ret = ''
        for i, line in enumerate(lines):
            ret += f'{self.line_range[0] + i}|{line}\n'
        return ret


def _create_chunks_from_raw_string(content: str, size: int) -> list[Chunk]:
    lines = content.split('\n')
    ret: list[Chunk] = []
    for i in range(0, len(lines), size):
        _cur_lines = lines[i : i + size]
        ret.append(
            Chunk(
                text='\n'.join(_cur_lines),
                line_range=(i + 1, i + len(_cur_lines)),
            )
        )
    return ret


def _collect_semantic_split_lines(
    node: Node,
    max_chunk_lines: int,
) -> set[int]:
    """Walk the AST to find 0-indexed line numbers where splitting is appropriate.

    A "split-after" line is the last line of a top-level AST child: splitting
    after it preserves the semantic unit. For oversized children, we recurse into
    their sub-children to discover finer split points.

    Args:
        node: A tree-sitter Node whose children define semantic units.
        max_chunk_lines: Target maximum lines per chunk. Nodes spanning more
            than this are recursed into.

    Returns:
        Set of 0-indexed line numbers where splitting *after* that line
        is semantically appropriate.
    """
    split_after: set[int] = set()
    if not node.children:
        return split_after

    for child in node.children:
        child_end_line = child.end_point[0]
        child_span = child_end_line - child.start_point[0] + 1

        # Mark the end of every child as a valid split point.
        split_after.add(child_end_line)

        # Recurse into children that are themselves oversized.
        if child_span > max_chunk_lines and child.children:
            split_after.update(_collect_semantic_split_lines(child, max_chunk_lines))

    return split_after


def _create_chunks_from_tree_sitter(
    tree: Tree,
    text: str,
    max_chunk_lines: int,
) -> list[Chunk]:
    """Create semantically-aware chunks from a tree-sitter parse tree.

    Args:
        tree: A tree_sitter.Tree returned by parser.parse().
        text: The original source text that was parsed.
        max_chunk_lines: Maximum number of lines per chunk.

    Returns:
        A list of Chunk objects covering the entire source text.
    """
    text_lines = text.split('\n')
    total_lines = len(text_lines)

    root = tree.root_node
    if not root.children:
        return _create_chunks_from_raw_string(text, max_chunk_lines)

    # Phase 1: discover where we prefer to split.
    split_after_set = _collect_semantic_split_lines(root, max_chunk_lines)

    # Phase 2: greedy line-based chunking with preferred split points.
    chunks: list[Chunk] = []
    chunk_start = 0  # 0-indexed, inclusive

    while chunk_start < total_lines:
        # The farthest line we could include without exceeding the budget.
        budget_end = min(chunk_start + max_chunk_lines - 1, total_lines - 1)

        if budget_end >= total_lines - 1:
            # Remaining lines fit in one chunk — emit and finish.
            chunk_text = '\n'.join(text_lines[chunk_start:total_lines])
            chunks.append(
                Chunk(
                    text=chunk_text,
                    line_range=(chunk_start + 1, total_lines),
                )
            )
            break

        # Search backward from budget_end for the latest semantic split point.
        best_split: int | None = None
        for candidate in range(budget_end, chunk_start - 1, -1):
            if candidate in split_after_set:
                best_split = candidate
                break

        if best_split is not None and best_split >= chunk_start:
            chunk_end = best_split
        else:
            # No semantic boundary found - raw split at budget limit.
            chunk_end = budget_end

        chunk_text = '\n'.join(text_lines[chunk_start : chunk_end + 1])
        chunks.append(
            Chunk(
                text=chunk_text,
                line_range=(chunk_start + 1, chunk_end + 1),
            )
        )
        chunk_start = chunk_end + 1

    return chunks


def create_chunks(
    text: str, size: int = 100, language: str | None = None
) -> list[Chunk]:
    if size <= 0:
        raise ValueError(f'size must be a positive integer, got {size}')

    try:
        parser = get_parser(language) if language is not None else None
    except (AttributeError, LookupError):
        logger.debug(f'Language {language} not supported. Falling back to raw string.')
        parser = None

    if parser is None:
        # fallback to raw string
        return _create_chunks_from_raw_string(text, size)

    return _create_chunks_from_tree_sitter(
        parser.parse(text.encode('utf-8')), text, max_chunk_lines=size
    )


def normalized_lcs(chunk: str, query: str) -> float:
    """Calculate the normalized Longest Common Subsequence (LCS) to compare file chunk with the query (e.g. edit draft).

    We normalize Longest Common Subsequence (LCS) by the length of the chunk
    to check how **much** of the chunk is covered by the query.
    """
    if len(chunk) == 0:
        return 0.0

    _score = LCSseq.similarity(chunk, query)

    return _score / len(chunk)


def get_top_k_chunk_matches(
    text: str, query: str, k: int = 3, max_chunk_size: int = 100
) -> list[Chunk]:
    """Get the top k chunks in the text that match the query.

    The query could be a string of draft code edits.

    Args:
        text: The text to search for the query.
        query: The query to search for in the text.
        k: The number of top chunks to return.
        max_chunk_size: The maximum number of lines in a chunk.
    """
    raw_chunks = create_chunks(text, max_chunk_size)
    chunks_with_lcs: list[Chunk] = [
        Chunk(
            text=chunk.text,
            line_range=chunk.line_range,
            normalized_lcs=normalized_lcs(chunk.text, query),
        )
        for chunk in raw_chunks
    ]
    sorted_chunks = sorted(
        chunks_with_lcs,
        key=lambda x: x.normalized_lcs if x.normalized_lcs is not None else 0.0,
        reverse=True,
    )
    return sorted_chunks[:k]
