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


def _lines_to_chunks(
    text_lines: list[str],
    start_line_idx: int,
    end_line_idx: int,
    max_chunk_lines: int,
) -> list[Chunk]:
    """Split a contiguous range of text lines into fixed-size chunks.

    This is a low-level helper used when padding lines (blank lines,
    comments between AST nodes, etc.) cannot be absorbed into an adjacent
    AST-backed chunk without exceeding max_chunk_lines.

    Args:
        text_lines: The full source text already split on '\\n'.
        start_line_idx: 0-indexed first line of the range (inclusive).
        end_line_idx: 0-indexed last line of the range (inclusive).
        max_chunk_lines: Maximum lines per emitted chunk.

    Returns:
        A list of Chunk objects with 1-indexed, inclusive line_range.
    """
    chunks: list[Chunk] = []
    for chunk_start in range(start_line_idx, end_line_idx + 1, max_chunk_lines):
        chunk_end = min(chunk_start + max_chunk_lines - 1, end_line_idx)
        chunk_text = '\n'.join(text_lines[chunk_start : chunk_end + 1])
        chunks.append(
            Chunk(
                text=chunk_text,
                line_range=(chunk_start + 1, chunk_end + 1),
            )
        )
    return chunks


def _chunk_sibling_nodes(
    nodes: list[Node],
    text_lines: list[str],
    max_chunk_lines: int,
    start_line_idx: int,
    end_line_idx: int,
) -> list[Chunk]:
    """Greedy-merge a list of sibling AST nodes into Chunks.

    Boundary expansion (absorbing blank lines / comments between AST
    nodes) is constrained by max_chunk_lines so that no emitted chunk
    silently exceeds the size limit.

    Args:
        nodes: Sibling tree-sitter Node objects to merge.
        text_lines: Source text already split on '\\n'.
        max_chunk_lines: Upper bound on lines per emitted chunk.
        start_line_idx: 0-indexed first line of the region (inclusive).
        end_line_idx: 0-indexed last line of the region (inclusive).

    Returns:
        A list of Chunk objects.
    """
    if not nodes:
        return _lines_to_chunks(
            text_lines, start_line_idx, end_line_idx, max_chunk_lines
        )

    # greedy-merge sibling nodes into groups that fit within
    # max_chunk_lines (measured by AST span only).
    groups: list[list[Node]] = []
    current_group: list[Node] = []
    current_group_start_line_idx: int = 0

    for node in nodes:
        node_start_line_idx: int = node.start_point[0]
        node_end_line_idx: int = node.end_point[0]

        if not current_group:
            current_group = [node]
            current_group_start_line_idx = node_start_line_idx
        else:
            merged_line_count = node_end_line_idx - current_group_start_line_idx + 1
            if merged_line_count <= max_chunk_lines:
                current_group.append(node)
            else:
                groups.append(current_group)
                current_group = [node]
                current_group_start_line_idx = node_start_line_idx

    if current_group:
        groups.append(current_group)

    # emit chunks, absorbing inter-group padding only when it
    # does not violate max_chunk_lines.
    result: list[Chunk] = []
    cursor: int = start_line_idx

    for i, group in enumerate(groups):
        group_ast_start: int = group[0].start_point[0]
        group_ast_end: int = group[-1].end_point[0]
        group_line_count: int = group_ast_end - group_ast_start + 1

        # The farthest line this group could claim (for suffix absorption).
        if i < len(groups) - 1:
            group_region_end: int = groups[i + 1][0].start_point[0] - 1
        else:
            group_region_end = end_line_idx

        # prefix absorption
        prefix_line_count: int = group_ast_start - cursor
        if prefix_line_count > 0:
            if prefix_line_count + group_line_count <= max_chunk_lines:
                chunk_start = cursor  # absorb prefix
            else:
                # Prefix too large
                result.extend(
                    _lines_to_chunks(
                        text_lines,
                        cursor,
                        group_ast_start - 1,
                        max_chunk_lines,
                    )
                )
                chunk_start = group_ast_start
        else:
            chunk_start = cursor

        # suffix / inter-group gap absorption
        current_chunk_line_count: int = group_ast_end - chunk_start + 1
        suffix_line_count: int = group_region_end - group_ast_end
        if (
            suffix_line_count > 0
            and current_chunk_line_count + suffix_line_count <= max_chunk_lines
        ):
            chunk_end = group_region_end
        else:
            chunk_end = group_ast_end

        # emit the chunk
        if len(group) == 1 and group_line_count > max_chunk_lines and group[0].children:
            # Single oversized node with children — recurse.
            sub_chunks = _chunk_sibling_nodes(
                nodes=group[0].children,
                text_lines=text_lines,
                max_chunk_lines=max_chunk_lines,
                start_line_idx=chunk_start,
                end_line_idx=chunk_end,
            )
            result.extend(sub_chunks)
        else:
            chunk_text = '\n'.join(text_lines[chunk_start : chunk_end + 1])
            result.append(
                Chunk(
                    text=chunk_text,
                    line_range=(chunk_start + 1, chunk_end + 1),
                )
            )

        cursor = chunk_end + 1

    # Handle any trailing lines not covered by the last group.
    if cursor <= end_line_idx:
        result.extend(
            _lines_to_chunks(text_lines, cursor, end_line_idx, max_chunk_lines)
        )

    return result


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
        A list of class Chunk objects covering the entire source text.
    """
    text_lines = text.split('\n')
    total_lines = len(text_lines)

    root = tree.root_node
    if not root.children:
        # Empty or un-parseable file – fall back to the naive splitter.
        return _create_chunks_from_raw_string(text, max_chunk_lines)

    return _chunk_sibling_nodes(
        nodes=root.children,
        text_lines=text_lines,
        max_chunk_lines=max_chunk_lines,
        start_line_idx=0,
        end_line_idx=total_lines - 1,
    )


def create_chunks(
    text: str, size: int = 100, language: str | None = None
) -> list[Chunk]:
    try:
        parser = get_parser(language) if language is not None else None
    except AttributeError:
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
