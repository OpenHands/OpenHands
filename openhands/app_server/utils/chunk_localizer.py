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


def _chunk_sibling_nodes(
    nodes: list[Node],
    text_lines: list[str],
    max_chunk_lines: int,
    start_line_idx: int,
    end_line_idx: int,
) -> list[Chunk]:
    """Greedy-merge a list of sibling AST nodes into Chunks.

    Args:
        nodes: Sibling tree-sitter "Node" objects to merge.
        text_lines: Source text already split on '\n'.
        max_chunk_lines: Upper bound on lines per emitted chunk.
        start_line_idx: 0-indexed first line index of the region these nodes occupy.
        end_line_idx: 0-indexed last line index of the region.

    Returns:
        A list of class Chunk objects.
    """
    if not nodes:
        # Defensive: emit the whole range as one chunk.
        chunk_text = '\n'.join(text_lines[start_line_idx : end_line_idx + 1])
        return [
            Chunk(
                text=chunk_text,
                line_range=(start_line_idx + 1, end_line_idx + 1),
            )
        ]

    groups: list[list[Node]] = []
    current_group: list[Node] = []
    current_group_start_line_idx: int = 0

    for node in nodes:
        # both are 0-indexed row
        node_start_line_idx: int = node.start_point[0]
        node_end_line_idx: int = node.end_point[0]

        if not current_group:
            current_group = [node]
            current_group_start_line_idx = node_start_line_idx
        else:
            # Lines the merged group would span
            merged_line_count = node_end_line_idx - current_group_start_line_idx + 1
            if merged_line_count <= max_chunk_lines:
                current_group.append(node)
            else:
                groups.append(current_group)
                current_group = [node]
                current_group_start_line_idx = node_start_line_idx

    if current_group:
        groups.append(current_group)

    result: list[Chunk] = []

    for i, group in enumerate(groups):
        group_ast_start_line_idx: int = group[0].start_point[0]
        group_ast_end_line_idx: int = group[-1].end_point[0]

        # Expand boundaries so blank lines are absorbed (full contiguity).
        actual_start_line_idx = start_line_idx if i == 0 else group_ast_start_line_idx
        if i < len(groups) - 1:
            next_group_ast_start_line_idx: int = groups[i + 1][0].start_point[0]
            actual_end_line_idx = next_group_ast_start_line_idx - 1
        else:
            actual_end_line_idx = end_line_idx

        # If a single node is oversized AND has children, recurse.
        if len(group) == 1:
            node = group[0]
            ast_line_count = group_ast_end_line_idx - group_ast_start_line_idx + 1
            if ast_line_count > max_chunk_lines and node.children:
                sub_chunks = _chunk_sibling_nodes(
                    nodes=node.children,
                    text_lines=text_lines,
                    max_chunk_lines=max_chunk_lines,
                    start_line_idx=actual_start_line_idx,
                    end_line_idx=actual_end_line_idx,
                )
                result.extend(sub_chunks)
                continue

        # Emit the group as a single Chunk.
        chunk_text = '\n'.join(
            text_lines[actual_start_line_idx : actual_end_line_idx + 1]
        )
        result.append(
            Chunk(
                text=chunk_text,
                line_range=(
                    actual_start_line_idx + 1,
                    actual_end_line_idx + 1,
                ),  # 1-indexed, inclusive
            )
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
