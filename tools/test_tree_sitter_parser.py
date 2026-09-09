"""tree-sitter wrapper never hard-depends on the package."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from parser_fallback import parse_source as fallback_parse_source  # noqa: E402


class TreeSitterParserTests(unittest.TestCase):
    def test_module_does_not_import_tree_sitter_at_load(self) -> None:
        source_path = os.path.join(TOOLS_DIR, "tree_sitter_parser.py")
        with open(source_path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertNotIn("import tree_sitter\n", text.split("def ", 1)[0])
        self.assertNotIn("from tree_sitter", text.split("def ", 1)[0])

    def test_missing_tree_sitter_falls_back(self) -> None:
        import tree_sitter_parser

        source = "def greet():\n    pass\n"
        with mock.patch.dict(sys.modules, {"tree_sitter": None}):
            with mock.patch(
                "tree_sitter_parser._try_tree_sitter",
                side_effect=lambda *args, **kwargs: (_ for _ in ()).throw(
                    ImportError("no tree_sitter")
                ),
            ):
                # Direct parse_source swallows ImportError inside _try_tree_sitter.
                pass
        with mock.patch(
            "tree_sitter_parser._try_tree_sitter",
            return_value=None,
        ):
            result = tree_sitter_parser.parse_source("a.py", source, "python")
        self.assertEqual(result, fallback_parse_source("a.py", source, "python"))

    def test_import_error_inside_try_falls_back(self) -> None:
        import tree_sitter_parser

        source = "def greet():\n    pass\n"
        real_import = __import__

        def boom(name, *args, **kwargs):
            if name == "tree_sitter":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=boom):
            result = tree_sitter_parser.parse_source("a.py", source, "python")
        self.assertEqual(
            {item["name"] for item in result["symbols"]},
            {item["name"] for item in fallback_parse_source("a.py", source, "python")["symbols"]},
        )


if __name__ == "__main__":
    unittest.main()
