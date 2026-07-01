import pytest

from openhands.app_server.utils.chunk_localizer import (
    Chunk,
    _create_chunks_from_raw_string,
    create_chunks,
    get_top_k_chunk_matches,
    normalized_lcs,
)


def assert_chunk_invariants(
    text: str,
    size: int,
    language: str | None = None,
    *,
    strict_max: bool = True,
) -> list[Chunk]:
    """Run ``create_chunks`` and assert all core invariants.

    Calls ``create_chunks(text, size, language)`` and checks:
        1. At least one chunk is produced
        2. First chunk starts at line 1
        3. Each chunk has start <= end  (no inverted ranges)
        4. Each chunk is within bounds  (end <= total_lines)
        5. Text line count matches declared range width
        6. Chunk.visualize() does not raise
        7. chunks[i].end + 1 == chunks[i+1].start  (contiguity)
        8. Last chunk ends at EOF
        9. '\\n'.join(c.text for c in chunks) == text  (reconstruction)
       10. (optional) every chunk has at most ``size`` lines

    Args:
        text: Source text to chunk.
        size: Maximum lines per chunk (passed to ``create_chunks``).
        language: Language hint passed to ``create_chunks``.
        strict_max: When True (default), also assert that no chunk
            exceeds ``size`` lines.

    Returns:
        The produced chunk list, so callers can make additional assertions.
    """
    chunks = create_chunks(text, size=size, language=language)
    N = len(text.split('\n'))

    assert len(chunks) >= 1, f'size={size}: expected at least one chunk, got 0'

    assert chunks[0].line_range[0] == 1, (
        f'size={size}: first chunk starts at {chunks[0].line_range[0]}, expected 1'
    )

    assert chunks[-1].line_range[1] == N, (
        f'size={size}: last chunk ends at {chunks[-1].line_range[1]}, expected {N}'
    )

    for i, c in enumerate(chunks):
        s, e = c.line_range
        assert s <= e, f'size={size}: chunk[{i}] has inverted range ({s}, {e})'
        assert 1 <= s <= N, (
            f'size={size}: chunk[{i}] start={s} is out of bounds [1, {N}]'
        )
        assert 1 <= e <= N, f'size={size}: chunk[{i}] end={e} is out of bounds [1, {N}]'
        declared = e - s + 1
        actual = len(c.text.split('\n'))
        assert actual == declared, (
            f'size={size}: chunk[{i}] range implies {declared} lines '
            f'but text has {actual}'
        )
        c.visualize()
        if strict_max:
            assert declared <= size, (
                f'size={size}: chunk[{i}] has {declared} lines, '
                f'exceeding limit: {c.line_range}'
            )

    for i in range(len(chunks) - 1):
        curr_end = chunks[i].line_range[1]
        next_start = chunks[i + 1].line_range[0]
        assert curr_end + 1 == next_start, (
            f'size={size}: gap/overlap between chunk[{i}] (ends={curr_end}) '
            f'and chunk[{i + 1}] (starts={next_start})'
        )

    reconstructed = '\n'.join(c.text for c in chunks)
    assert reconstructed == text, (
        f'size={size}: reconstruction failed.\n'
        f'  expected : {repr(text[:120])}\n'
        f'  got      : {repr(reconstructed[:120])}'
    )

    return chunks


def test_chunk_creation():
    chunk = Chunk(text='test chunk', line_range=(1, 1))
    assert chunk.text == 'test chunk'
    assert chunk.line_range == (1, 1)
    assert chunk.normalized_lcs is None


def test_chunk_visualization(capsys):
    chunk = Chunk(text='line1\nline2', line_range=(1, 2))
    assert chunk.visualize() == '1|line1\n2|line2\n'


def test_chunk_visualization_with_special_characters():
    chunk = Chunk(text='line1\nline2\t\nline3\r', line_range=(1, 3))
    assert chunk.visualize() == '1|line1\n2|line2\t\n3|line3\r\n'


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


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 4])
def test_create_chunks_different_sizes(chunk_size):
    text = 'line1\nline2\nline3\nline4'
    chunks = create_chunks(text, size=chunk_size)
    assert len(chunks) == (4 + chunk_size - 1) // chunk_size
    assert sum(len(chunk.text.split('\n')) for chunk in chunks) == 4


def test_normalized_lcs():
    chunk = 'abcdef'
    edit_draft = 'abcxyz'
    assert normalized_lcs(chunk, edit_draft) == 0.5


def test_normalized_lcs_edge_cases():
    assert normalized_lcs('', '') == 0.0
    assert normalized_lcs('a', '') == 0.0
    assert normalized_lcs('', 'a') == 0.0
    assert normalized_lcs('abcde', 'ace') == 0.6


def test_normalized_lcs_with_unicode():
    chunk = 'Hello, 世界!'
    edit_draft = 'Hello, world!'
    assert 0 < normalized_lcs(chunk, edit_draft) < 1


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
    assert len(matches) == 3


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


def test_create_chunks_unsupported_language_fallback():
    """Unsupported language falls back to raw string chunking."""
    text = 'line1\nline2\nline3\nline4'
    chunks = create_chunks(text, size=2, language='brainfuck_not_real')
    assert len(chunks) == 2
    assert chunks[0].text == 'line1\nline2'
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].text == 'line3\nline4'
    assert chunks[1].line_range == (3, 4)


@pytest.mark.parametrize('size', [0, -1, -100])
@pytest.mark.parametrize('language', [None, 'python'])
def test_create_chunks_non_positive_size_raises(size, language):
    """A non-positive size must fail fast rather than loop forever.

    The tree-sitter path would otherwise spin indefinitely (the chunk cursor
    never advances when the budget is empty), so guard it explicitly.
    """
    with pytest.raises(ValueError):
        create_chunks('def foo():\n    pass', size=size, language=language)


def test_create_chunks_no_language_uses_raw():
    """When language=None the raw string chunker is used."""
    text = 'a\nb\nc\nd\ne'
    chunks = create_chunks(text, size=2, language=None)
    assert len(chunks) == 3
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].line_range == (3, 4)
    assert chunks[2].line_range == (5, 5)


def test_create_chunks_empty_file():
    chunks = create_chunks('', size=10, language='python')
    assert len(chunks) == 1
    assert chunks[0].text == ''
    assert chunks[0].line_range == (1, 1)


def test_create_chunks_empty_file_raw():
    chunks = create_chunks('', size=10)
    assert len(chunks) == 1
    assert chunks[0].text == ''
    assert chunks[0].line_range == (1, 1)


def test_create_chunks_tree_sitter_whitespace_only():
    text = '\n\n\n\n\n'
    assert_chunk_invariants(text, size=2, language='python')


def test_create_chunks_tree_sitter_python_basic():
    text = """\n    def foo():\n        print("foo")\n    def bar():\n        print("bar")\n    """
    chunks = create_chunks(text, size=3, language='python')
    assert len(chunks) == 2
    assert chunks[0].line_range == (1, 3)
    assert 'def foo():' in chunks[0].text
    assert chunks[1].line_range == (4, 6)
    assert 'def bar():' in chunks[1].text


def test_create_chunks_tree_sitter_python_oversized():
    text = """\n    class MyClass:\n        def method1(self):\n            a = 1\n            b = 2\n\n        def method2(self):\n            c = 3\n            d = 4\n    """
    assert_chunk_invariants(text, size=4, language='python')


def test_create_chunks_tree_sitter_prefix_respects_max_lines():
    """Lines before the first AST node must not push a chunk past max_chunk_lines."""
    text = (
        '# comment 1\n# comment 2\n# comment 3\n'
        '# comment 4\n# comment 5\ndef foo():\n    pass'
    )
    assert_chunk_invariants(text, size=3, language='python')


def test_create_chunks_tree_sitter_gap_respects_max_lines():
    """Blank lines between AST nodes must not push a chunk past max_chunk_lines."""
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
    assert_chunk_invariants(text, size=3, language='python')


def test_create_chunks_tree_sitter_suffix_respects_max_lines():
    """Trailing lines after the last AST node must not push a chunk past max_chunk_lines."""
    text = 'def foo():\n    pass\n\n\n\n\n'
    assert_chunk_invariants(text, size=3, language='python')


def test_create_chunks_tree_sitter_single_line_functions():
    """Multiple single-line statements are grouped up to max_chunk_lines."""
    text = 'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6'
    assert_chunk_invariants(text, size=2, language='python')


def test_create_chunks_tree_sitter_tuple_literal_size_5():
    text = 'DATA = tuple([\n' + '\n'.join(f'    {i},' for i in range(12)) + '\n])'
    assert_chunk_invariants(text, size=5, language='python')


def test_create_chunks_tree_sitter_tuple_literal_size_100():
    text = 'DATA = tuple([\n' + '\n'.join(f'    {i},' for i in range(12)) + '\n])'
    chunks = assert_chunk_invariants(text, size=100, language='python')
    assert len(chunks) == 1


def test_create_chunks_tree_sitter_large_tuple_literal():
    text = 'DATA = tuple([\n' + '\n'.join(f'    {i},' for i in range(200)) + '\n])'
    assert_chunk_invariants(text, size=100, language='python')


def test_create_chunks_tree_sitter_deeply_nested():
    """Deeply nested AST produces valid chunks at small size."""
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
    assert_chunk_invariants(text, size=3, language='python')


def test_create_chunks_tree_sitter_single_huge_function():
    """A function longer than max_chunk_lines is still chunked correctly."""
    body_lines = [f'    x{i} = {i}' for i in range(20)]
    text = 'def big():\n' + '\n'.join(body_lines)
    assert_chunk_invariants(text, size=5, language='python')


def test_create_chunks_tree_sitter_same_row_nested_nodes():
    """Nodes sharing the same start row must not produce inverted or duplicate ranges."""
    text = 'x = [1, 2, 3]'
    assert_chunk_invariants(text, size=1, language='python')


def test_create_chunks_tree_sitter_multiline_dict():
    """Multi-line dict literal with a same-row opening brace."""
    text = 'config = {\n' + '\n'.join(f'    "key{i}": {i},' for i in range(15)) + '\n}'
    assert_chunk_invariants(text, size=5, language='python')


def test_create_chunks_tree_sitter_nested_function_calls():
    """Deeply nested function calls on a single line."""
    text = 'result = foo(bar(baz(qux(42))))'
    assert_chunk_invariants(text, size=1, language='python')


def test_create_chunks_tree_sitter_mixed_constructs():
    """File mixing imports, decorators, functions, and classes."""
    text = '\n'.join(
        [
            'import os',
            'import sys',
            '',
            '',
            '@decorator',
            'def helper():',
            '    return 42',
            '',
            '',
            'class MyClass:',
            '    """A docstring."""',
            '',
            '    def method(self):',
            '        pass',
            '',
            '    @staticmethod',
            '    def static_method():',
            '        return 1',
        ]
    )
    assert_chunk_invariants(text, size=5, language='python')


def test_create_chunks_tree_sitter_visualize_all_chunks():
    """Chunk.visualize() must not raise for any chunk produced by the tree-sitter path."""
    text = 'DATA = tuple([\n' + '\n'.join(f'    {i},' for i in range(12)) + '\n])'
    for size in [1, 2, 3, 5, 10, 100]:
        chunks = create_chunks(text, size=size, language='python')
        for c in chunks:
            c.visualize()


_CODE_SAMPLES: dict[str, str] = {
    'tuple_literal': (
        'DATA = tuple([\n' + '\n'.join(f'    {i},' for i in range(12)) + '\n])'
    ),
    'large_tuple': (
        'DATA = tuple([\n' + '\n'.join(f'    {i},' for i in range(200)) + '\n])'
    ),
    'functions': '\n'.join(
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
    ),
    'deeply_nested': '\n'.join(
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
    ),
    'single_huge_function': (
        'def big():\n' + '\n'.join(f'    x{i} = {i}' for i in range(50))
    ),
    'single_line': 'x = 42',
    'empty': '',
    'whitespace_only': '\n\n\n\n\n',
    'assignments': 'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6',
    'multiline_string': 'x = """\nline1\nline2\nline3\nline4\nline5\n"""',
    'list_comprehension': (
        'result = [\n' + '\n'.join(f'    item_{i},' for i in range(20)) + '\n]'
    ),
}


@pytest.mark.parametrize('size', [1, 2, 3, 5, 10, 100])
@pytest.mark.parametrize('sample_name', list(_CODE_SAMPLES.keys()))
def test_invariants_parametrized(sample_name: str, size: int):
    """Invariant sweep across multiple code samples and chunk sizes."""
    text = _CODE_SAMPLES[sample_name]
    assert_chunk_invariants(text, size=size, language='python')


@pytest.mark.parametrize('size', [1, 2, 3, 5, 10])
def test_max_chunk_lines_enforced(size):
    """max_chunk_lines is strictly respected."""
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
    assert_chunk_invariants(text, size=size, language='python')


class TestChunkInvariantPython:
    """Invariant tests for the tree-sitter chunking path."""

    SIZES = [1, 2, 3, 5, 7, 10, 15, 50, 100]

    def _check_all_sizes(self, text: str) -> None:
        for size in self.SIZES:
            assert_chunk_invariants(text, size=size, language='python')

    def test_tuple_literal_all_sizes(self):
        """Multi-line tuple literal with nested same-row AST nodes."""
        text = 'DATA = tuple([\n' + '\n'.join(f'    {i},' for i in range(12)) + '\n])'
        self._check_all_sizes(text)

    def test_flat_functions(self):
        """Multiple top-level functions."""
        text = '\n'.join(
            [
                'def foo():',
                '    return 1',
                '',
                'def bar():',
                '    return 2',
                '',
                'def baz():',
                '    return 3',
            ]
        )
        self._check_all_sizes(text)

    def test_single_oversized_function(self):
        """Single function whose body exceeds max_chunk_lines."""
        body = '\n'.join(f'    x_{i} = {i}' for i in range(30))
        text = f'def big_function():\n{body}\n    return x_0'
        self._check_all_sizes(text)

    def test_class_with_many_methods(self):
        """Class whose total size exceeds max_chunk_lines."""
        methods = []
        for i in range(8):
            methods.append(f'    def method_{i}(self):')
            methods.append(f'        return {i}')
            methods.append('')
        text = 'class MyClass:\n' + '\n'.join(methods)
        self._check_all_sizes(text)

    def test_deeply_nested_class(self):
        """Nested class definition forcing multi-level descent."""
        text = (
            'class Outer:\n'
            '    class Inner:\n'
            + '\n'.join(f'        def m{i}(self): return {i}' for i in range(10))
            + '\n    def outer_method(self):\n        pass\n'
        )
        self._check_all_sizes(text)

    def test_leading_and_trailing_blank_lines(self):
        """Blank lines before the first and after the last AST node."""
        text = '\n\n\ndef foo():\n    return 1\n\n\n'
        self._check_all_sizes(text)

    def test_same_row_siblings(self):
        """Multiple top-level statements, some spanning multiple lines."""
        text = (
            'x = [1, 2, 3]\n'
            "y = {'a': 1, 'b': 2}\n"
            'z = (i for i in range(100))\n'
            'DATA = tuple([\n' + '\n'.join(f'    {i},' for i in range(8)) + '\n])\n'
            'W = list(range(20))\n'
        )
        self._check_all_sizes(text)

    def test_empty_file(self):
        """Empty input produces exactly one chunk."""
        chunks = create_chunks('', size=10, language='python')
        assert len(chunks) == 1
        assert chunks[0].text == ''

    def test_single_line(self):
        """Single-line file produces exactly one chunk regardless of size."""
        text = 'x = 1'
        self._check_all_sizes(text)

    def test_large_file_default_size(self):
        """Large list literal followed by multiple functions."""
        big_list = (
            'QUERIES = [\n'
            + '\n'.join(f"    'query_{i}'," for i in range(120))
            + '\n]\n'
        )
        functions = '\n'.join(
            f'def handler_{i}(ctx):\n    return QUERIES[{i}]\n' for i in range(10)
        )
        text = big_list + '\n' + functions
        assert_chunk_invariants(text, size=100, language='python')
        assert_chunk_invariants(text, size=50, language='python')
        assert_chunk_invariants(text, size=10, language='python')

    def test_fallback_for_unsupported_language(self):
        """Unsupported language falls back to raw-string chunking."""
        text = 'line one\nline two\nline three\nline four\nline five\n'
        assert_chunk_invariants(text, size=2, language='brainfuck')

    def test_fallback_for_none_language(self):
        """language=None uses raw-string chunking."""
        text = '\n'.join(f'line {i}' for i in range(20))
        assert_chunk_invariants(text, size=5, language=None)


class TestSemanticBoundaries:
    """Assert that tree-sitter produces *different* boundaries than raw splitting.

    Each test constructs input where semantic chunking and raw line splitting
    produce different chunk boundaries, then asserts the semantic boundaries
    explicitly.  If ``_create_chunks_from_tree_sitter`` were swapped for the
    raw splitter, these tests would fail.
    """

    def test_unequal_sibling_functions_split_on_function_boundary(self):
        """A short function followed by a longer one: chunk ends at the function
        boundary, not at the raw budget line.
        """
        text = '\n'.join(
            [
                'def short():',
                '    return 1',
                '',
                'def long_func():',
                '    a = 1',
                '    b = 2',
                '    return a + b',
            ]
        )
        chunks = create_chunks(text, size=5, language='python')

        # Semantic boundary: first chunk contains only short() (lines 1-2),
        # second chunk contains the blank line + long_func() (lines 3-7).
        boundaries = [c.line_range for c in chunks]
        assert boundaries == [(1, 2), (3, 7)], (
            f'Expected semantic split after short(), got {boundaries}'
        )

        # Verify this differs from raw splitting.
        raw_chunks = _create_chunks_from_raw_string(text, 5)
        raw_boundaries = [c.line_range for c in raw_chunks]
        assert raw_boundaries != boundaries, (
            'Semantic and raw boundaries should differ for this input'
        )

    def test_oversized_class_splits_between_methods(self):
        """A class with three methods: chunks split between methods, not mid-method."""
        text = '\n'.join(
            [
                'class MyClass:',
                '    def method_a(self):',
                '        return 1',
                '',
                '    def method_b(self):',
                '        return 2',
                '',
                '    def method_c(self):',
                '        return 3',
            ]
        )
        chunks = create_chunks(text, size=4, language='python')

        boundaries = [c.line_range for c in chunks]
        assert boundaries == [(1, 3), (4, 6), (7, 9)], (
            f'Expected splits between methods, got {boundaries}'
        )

        raw_chunks = _create_chunks_from_raw_string(text, 4)
        raw_boundaries = [c.line_range for c in raw_chunks]
        assert raw_boundaries != boundaries

    def test_semantic_split_keeps_decorator_with_function(self):
        """A decorated function stays in the same chunk as its decorator."""
        text = '\n'.join(
            [
                'import os',
                'import sys',
                '',
                '@my_decorator',
                'def decorated():',
                '    return 42',
                '',
            ]
        )
        chunks = create_chunks(text, size=5, language='python')
        boundaries = [c.line_range for c in chunks]

        # The decorator + function must be in the same chunk.
        # Find the chunk containing '@my_decorator'.
        decorator_chunk = [c for c in chunks if '@my_decorator' in c.text]
        assert len(decorator_chunk) == 1
        assert 'def decorated():' in decorator_chunk[0].text, (
            'Decorator and function definition should be in the same chunk'
        )
        assert 'return 42' in decorator_chunk[0].text, (
            'Function body should be in the same chunk as its decorator'
        )

        # Verify this differs from raw.
        raw_chunks = _create_chunks_from_raw_string(text, 5)
        raw_boundaries = [c.line_range for c in raw_chunks]
        assert raw_boundaries != boundaries
