import pytest

from openhands.app_server.utils.chunk_localizer import (
    Chunk,
    create_chunks,
    get_top_k_chunk_matches,
    normalized_lcs,
)


def test_chunk_creation():
    chunk = Chunk(text='test chunk', line_range=(1, 1))
    assert chunk.text == 'test chunk'
    assert chunk.line_range == (1, 1)
    assert chunk.normalized_lcs is None


def test_chunk_visualization(capsys):
    chunk = Chunk(text='line1\nline2', line_range=(1, 2))
    assert chunk.visualize() == '1|line1\n2|line2\n'


def test_create_chunks_raw_string():
    text = 'line1\nline2\nline3\nline4\nline5'
    chunks = create_chunks(text, size=2)
    assert len(chunks) == 3
    assert chunks[0].text == 'line1\nline2'
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].text == 'line3\nline4'
    assert chunks[1].line_range == (3, 4)
    assert chunks[2].text == 'line5'
    assert chunks[2].line_range == (5, 5)


def test_normalized_lcs():
    chunk = 'abcdef'
    edit_draft = 'abcxyz'
    assert normalized_lcs(chunk, edit_draft) == 0.5


def test_get_top_k_chunk_matches():
    text = 'chunk1\nchunk2\nchunk3\nchunk4'
    query = 'chunk2'
    matches = get_top_k_chunk_matches(text, query, k=2, max_chunk_size=1)
    assert len(matches) == 2
    assert matches[0].text == 'chunk2'
    assert matches[0].line_range == (2, 2)
    assert matches[0].normalized_lcs == 1.0
    assert matches[1].text == 'chunk1'
    assert matches[1].line_range == (1, 1)
    assert matches[1].normalized_lcs == 5 / 6
    assert matches[0].normalized_lcs > matches[1].normalized_lcs


def test_create_chunks_with_empty_lines():
    text = 'line1\n\nline3\n\n\nline6'
    chunks = create_chunks(text, size=2)
    assert len(chunks) == 3
    assert chunks[0].text == 'line1\n'
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].text == 'line3\n'
    assert chunks[1].line_range == (3, 4)
    assert chunks[2].text == '\nline6'
    assert chunks[2].line_range == (5, 6)


def test_create_chunks_with_large_size():
    text = 'line1\nline2\nline3'
    chunks = create_chunks(text, size=10)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].line_range == (1, 3)


def test_create_chunks_with_last_chunk_smaller():
    text = 'line1\nline2\nline3'
    chunks = create_chunks(text, size=2)
    assert len(chunks) == 2
    assert chunks[0].text == 'line1\nline2'
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].text == 'line3'
    assert chunks[1].line_range == (3, 3)


def test_normalized_lcs_edge_cases():
    assert normalized_lcs('', '') == 0.0
    assert normalized_lcs('a', '') == 0.0
    assert normalized_lcs('', 'a') == 0.0
    assert normalized_lcs('abcde', 'ace') == 0.6


def test_get_top_k_chunk_matches_with_ties():
    text = 'chunk1\nchunk2\nchunk3\nchunk1'
    query = 'chunk'
    matches = get_top_k_chunk_matches(text, query, k=3, max_chunk_size=1)
    assert len(matches) == 3
    assert all(match.normalized_lcs == 5 / 6 for match in matches)
    assert {match.text for match in matches} == {'chunk1', 'chunk2', 'chunk3'}


def test_get_top_k_chunk_matches_with_large_k():
    text = 'chunk1\nchunk2\nchunk3'
    query = 'chunk'
    matches = get_top_k_chunk_matches(text, query, k=10, max_chunk_size=1)
    assert len(matches) == 3  # Should return all chunks even if k is larger


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 4])
def test_create_chunks_different_sizes(chunk_size):
    text = 'line1\nline2\nline3\nline4'
    chunks = create_chunks(text, size=chunk_size)
    assert len(chunks) == (4 + chunk_size - 1) // chunk_size
    assert sum(len(chunk.text.split('\n')) for chunk in chunks) == 4


def test_chunk_visualization_with_special_characters():
    chunk = Chunk(text='line1\nline2\t\nline3\r', line_range=(1, 3))
    assert chunk.visualize() == '1|line1\n2|line2\t\n3|line3\r\n'


def test_normalized_lcs_with_unicode():
    chunk = 'Hello, 世界!'
    edit_draft = 'Hello, world!'
    assert 0 < normalized_lcs(chunk, edit_draft) < 1


def test_get_top_k_chunk_matches_with_overlapping_chunks():
    text = 'chunk1\nchunk2\nchunk3\nchunk4'
    query = 'chunk2\nchunk3'
    matches = get_top_k_chunk_matches(text, query, k=2, max_chunk_size=2)
    assert len(matches) == 2
    assert matches[0].text == 'chunk1\nchunk2'
    assert matches[0].line_range == (1, 2)
    assert matches[1].text == 'chunk3\nchunk4'
    assert matches[1].line_range == (3, 4)
    assert matches[0].normalized_lcs == matches[1].normalized_lcs


def test_create_chunks_tree_sitter_python_basic():
    text = """
    def foo():
        print("foo")
    def bar():
        print("bar")
    """
    chunks = create_chunks(text, size=3, language='python')
    assert len(chunks) == 2
    assert chunks[0].line_range == (1, 3)
    assert 'def foo():' in chunks[0].text
    assert chunks[1].line_range == (4, 6)
    assert 'def bar():' in chunks[1].text


def test_create_chunks_tree_sitter_python_oversized():
    text = """
    class MyClass:
        def method1(self):
            a = 1
            b = 2

        def method2(self):
            c = 3
            d = 4
    """
    chunks = create_chunks(text, size=4, language='python')
    assert len(chunks) > 0
    # Check full contiguity
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]

    assert chunks[0].line_range[0] == 1
    assert chunks[-1].line_range[1] == len(text.split('\n'))


def test_create_chunks_tree_sitter_prefix_respects_max_lines():
    """Prefix lines before the first AST node must not cause chunks to exceed max_chunk_lines."""
    text = (
        '# comment 1\n# comment 2\n# comment 3\n'
        '# comment 4\n# comment 5\ndef foo():\n    pass'
    )
    chunks = create_chunks(text, size=3, language='python')
    # Every chunk must respect the size constraint.
    for chunk in chunks:
        line_count = chunk.line_range[1] - chunk.line_range[0] + 1
        assert line_count <= 3, f'Chunk exceeded max size: {chunk.line_range}'
    # Full contiguity.
    assert chunks[0].line_range[0] == 1
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]
    assert chunks[-1].line_range[1] == len(text.split('\n'))


def test_create_chunks_tree_sitter_gap_respects_max_lines():
    """Inter-group gaps must not cause chunks to exceed max_chunk_lines."""
    lines = [
        'def foo():',
        '    pass',
        '',
        '',
        '',
        '',
        '',
        'def bar():',
        '    pass',
    ]
    text = '\n'.join(lines)
    chunks = create_chunks(text, size=3, language='python')
    for chunk in chunks:
        line_count = chunk.line_range[1] - chunk.line_range[0] + 1
        assert line_count <= 3, f'Chunk exceeded max size: {chunk.line_range}'
    # Full contiguity.
    assert chunks[0].line_range[0] == 1
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]
    assert chunks[-1].line_range[1] == len(lines)


def test_create_chunks_tree_sitter_suffix_respects_max_lines():
    """Trailing lines after the last AST node must not cause chunks to exceed max_chunk_lines."""
    text = 'def foo():\n    pass\n\n\n\n\n'
    chunks = create_chunks(text, size=3, language='python')
    for chunk in chunks:
        line_count = chunk.line_range[1] - chunk.line_range[0] + 1
        assert line_count <= 3, f'Chunk exceeded max size: {chunk.line_range}'
    # Full contiguity.
    assert chunks[0].line_range[0] == 1
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]
    assert chunks[-1].line_range[1] == len(text.split('\n'))


def test_create_chunks_unsupported_language_fallback():
    """Unsupported language falls back to raw string chunking."""
    text = 'line1\nline2\nline3\nline4'
    chunks = create_chunks(text, size=2, language='brainfuck_not_real')
    # Should produce the same result as no-language (raw) chunking.
    assert len(chunks) == 2
    assert chunks[0].text == 'line1\nline2'
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].text == 'line3\nline4'
    assert chunks[1].line_range == (3, 4)


def test_create_chunks_no_language_uses_raw():
    """When language=None the raw string chunker is used."""
    text = 'a\nb\nc\nd\ne'
    chunks = create_chunks(text, size=2, language=None)
    assert len(chunks) == 3
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].line_range == (3, 4)
    assert chunks[2].line_range == (5, 5)


def test_create_chunks_empty_file():
    """An empty string should produce a single empty chunk."""
    chunks = create_chunks('', size=10, language='python')
    assert len(chunks) == 1
    assert chunks[0].text == ''
    assert chunks[0].line_range == (1, 1)


def test_create_chunks_empty_file_raw():
    """An empty string with raw chunking should produce a single empty chunk."""
    chunks = create_chunks('', size=10)
    assert len(chunks) == 1
    assert chunks[0].text == ''
    assert chunks[0].line_range == (1, 1)


def test_create_chunks_tree_sitter_deeply_nested():
    """Deeply nested code should recurse into children and still produce valid chunks."""
    text = '\n'.join(
        [
            'class Outer:',
            '    class Inner:',
            '        def method(self):',
            '            if True:',
            '                for i in range(10):',
            '                    x = i',
            '                    y = i + 1',
            '                    z = i + 2',
        ]
    )
    chunks = create_chunks(text, size=3, language='python')
    assert len(chunks) > 0
    # Every chunk must respect the size constraint.
    for chunk in chunks:
        line_count = chunk.line_range[1] - chunk.line_range[0] + 1
        assert line_count <= 3, f'Chunk exceeded max size: {chunk.line_range}'
    # Full contiguity.
    assert chunks[0].line_range[0] == 1
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]
    assert chunks[-1].line_range[1] == len(text.split('\n'))


def test_create_chunks_tree_sitter_single_huge_function():
    """A single large function with no child structure that can be split further.

    When the AST node is oversized but has only leaf children (expressions),
    it should still be emitted without crashing, even if the chunk exceeds
    max_chunk_lines (no further AST split is possible).
    """
    body_lines = [f'    x{i} = {i}' for i in range(20)]
    text = 'def big():\n' + '\n'.join(body_lines)
    chunks = create_chunks(text, size=5, language='python')
    assert len(chunks) > 0
    # Full contiguity.
    assert chunks[0].line_range[0] == 1
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]
    assert chunks[-1].line_range[1] == len(text.split('\n'))
    # All text is covered.
    reconstructed = '\n'.join(chunk.text for chunk in chunks)
    assert reconstructed == text


@pytest.mark.parametrize('size', [1, 2, 3, 5, 10])
def test_create_chunks_tree_sitter_max_chunk_lines_enforced(size):
    """max_chunk_lines must be respected for various chunk sizes."""
    text = '\n'.join(
        [
            'import os',
            'import sys',
            '',
            'def foo():',
            '    return 1',
            '',
            'def bar():',
            '    return 2',
            '',
            'class Baz:',
            '    def method(self):',
            '        pass',
        ]
    )
    chunks = create_chunks(text, size=size, language='python')
    assert len(chunks) > 0
    for chunk in chunks:
        line_count = chunk.line_range[1] - chunk.line_range[0] + 1
        # Allow leaf AST nodes that are inherently larger than size,
        # but padding-expanded chunks must not exceed size.
        assert line_count <= max(size, 3), (
            f'Chunk {chunk.line_range} has {line_count} lines, '
            f'expected at most {max(size, 3)}'
        )
    # Full contiguity.
    assert chunks[0].line_range[0] == 1
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]
    assert chunks[-1].line_range[1] == len(text.split('\n'))


def test_create_chunks_tree_sitter_single_line_functions():
    """Multiple single-line statements should be grouped up to max_chunk_lines."""
    text = 'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6'
    chunks = create_chunks(text, size=2, language='python')
    assert len(chunks) > 0
    for chunk in chunks:
        line_count = chunk.line_range[1] - chunk.line_range[0] + 1
        assert line_count <= 2
    # Full contiguity.
    assert chunks[0].line_range[0] == 1
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]
    assert chunks[-1].line_range[1] == 6


def test_create_chunks_tree_sitter_whitespace_only():
    """A file containing only whitespace should chunk without errors."""
    text = '\n\n\n\n\n'
    chunks = create_chunks(text, size=2, language='python')
    assert len(chunks) > 0
    # Full contiguity.
    assert chunks[0].line_range[0] == 1
    for i in range(len(chunks) - 1):
        assert chunks[i].line_range[1] + 1 == chunks[i + 1].line_range[0]
    assert chunks[-1].line_range[1] == len(text.split('\n'))
