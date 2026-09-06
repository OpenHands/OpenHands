"""Deterministic lexer-based parsers for Python, JS, TS, and Markdown."""

from __future__ import annotations

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from parser_fallback import (  # noqa: E402
    KIND_CLASS,
    KIND_FUNCTION,
    KIND_METHOD,
    KIND_VARIABLE,
    detect_language,
    parse_source,
)

PY_FIXTURE = '''\
import os
from helper import greet as say_hi
from localmod import ping

COUNT = 1

class Greeter:
    def hello(self):
        say_hi()
        ping()

def greet():
    print("hi")
    os.path.join("a", "b")
'''

JS_FIXTURE = '''\
import { foo } from "./lib";
const bar = require("./util");

export function run() {
  foo();
  bar();
}

export class App {
  start() {
    run();
  }
}

export const LIMIT = 10;
'''

TS_FIXTURE = '''\
import { User } from "./user";

export interface Person {
  name: string;
}

export type Id = string;

export function greet(person: Person): Id {
  return person.name;
}

export class Roster {
  add(user: User) {
    greet({ name: user.name });
  }
}
'''

MD_FIXTURE = '''\
# Title

See [helper](./helper.py) and [docs](https://example.com).

Call `greet()` from the intro.
'''


def _names(items: list[dict], key: str = "name") -> set[str]:
    return {item[key] for item in items}


class DetectLanguageTests(unittest.TestCase):
    def test_extension_map(self) -> None:
        self.assertEqual(detect_language("a.py"), "python")
        self.assertEqual(detect_language("a.js"), "javascript")
        self.assertEqual(detect_language("a.tsx"), "typescript")
        self.assertEqual(detect_language("a.md"), "markdown")
        self.assertIsNone(detect_language("a.txt"))


class PythonParserTests(unittest.TestCase):
    def test_extracts_symbols_imports_defs_and_calls(self) -> None:
        result = parse_source("mod.py", PY_FIXTURE, "python")
        symbols = {(item["name"], item["kind"]) for item in result["symbols"]}
        self.assertIn(("Greeter", KIND_CLASS), symbols)
        self.assertIn(("hello", KIND_METHOD), symbols)
        self.assertIn(("greet", KIND_FUNCTION), symbols)
        self.assertIn(("COUNT", KIND_VARIABLE), symbols)
        self.assertEqual(
            _names(result["imports"]),
            {"os", "helper", "localmod"},
        )
        self.assertIn("say_hi", _names(result["calls"]))
        self.assertIn("ping", _names(result["calls"]))
        self.assertIn("print", _names(result["calls"]))
        self.assertTrue(all("start_line" in item for item in result["symbols"]))

    def test_is_deterministic(self) -> None:
        a = parse_source("mod.py", PY_FIXTURE, "python")
        b = parse_source("mod.py", PY_FIXTURE, "python")
        self.assertEqual(a, b)


class JavascriptParserTests(unittest.TestCase):
    def test_extracts_esm_cjs_defs_and_calls(self) -> None:
        result = parse_source("app.js", JS_FIXTURE, "javascript")
        symbols = {(item["name"], item["kind"]) for item in result["symbols"]}
        self.assertIn(("run", KIND_FUNCTION), symbols)
        self.assertIn(("App", KIND_CLASS), symbols)
        self.assertIn(("start", KIND_METHOD), symbols)
        self.assertIn(("LIMIT", KIND_VARIABLE), symbols)
        self.assertEqual(_names(result["imports"]), {"./lib", "./util"})
        self.assertIn("foo", _names(result["calls"]))
        self.assertIn("bar", _names(result["calls"]))
        self.assertIn("run", _names(result["calls"]))

    def test_is_deterministic(self) -> None:
        a = parse_source("app.js", JS_FIXTURE, "javascript")
        b = parse_source("app.js", JS_FIXTURE, "javascript")
        self.assertEqual(a, b)


class TypescriptParserTests(unittest.TestCase):
    def test_extracts_types_and_references(self) -> None:
        result = parse_source("app.ts", TS_FIXTURE, "typescript")
        names = _names(result["symbols"])
        self.assertIn("Person", names)
        self.assertIn("Id", names)
        self.assertIn("greet", names)
        self.assertIn("Roster", names)
        self.assertIn("add", names)
        self.assertEqual(_names(result["imports"]), {"./user"})
        self.assertIn("greet", _names(result["calls"]))
        self.assertIn("Person", _names(result["references"]))
        self.assertIn("User", _names(result["references"]))

    def test_is_deterministic(self) -> None:
        a = parse_source("app.ts", TS_FIXTURE, "typescript")
        b = parse_source("app.ts", TS_FIXTURE, "typescript")
        self.assertEqual(a, b)


class MarkdownParserTests(unittest.TestCase):
    def test_extracts_headings_links_and_call_sites(self) -> None:
        result = parse_source("README.md", MD_FIXTURE, "markdown")
        self.assertIn("Title", _names(result["symbols"]))
        self.assertEqual(_names(result["imports"]), {"./helper.py", "https://example.com"})
        self.assertIn("greet", _names(result["calls"]))

    def test_is_deterministic(self) -> None:
        a = parse_source("README.md", MD_FIXTURE, "markdown")
        b = parse_source("README.md", MD_FIXTURE, "markdown")
        self.assertEqual(a, b)


class LanguageAgnosticSmokeTests(unittest.TestCase):
    def test_empty_source_is_stable(self) -> None:
        for language in ("python", "javascript", "typescript", "markdown"):
            result = parse_source("x", "", language)
            self.assertEqual(result["symbols"], [])
            self.assertEqual(result["imports"], [])
            self.assertEqual(result["calls"], [])
            self.assertEqual(result["language"], language)

    def test_unknown_language_returns_empty(self) -> None:
        result = parse_source("a.rs", "fn main() {}", "rust")
        self.assertEqual(result["symbols"], [])
        self.assertEqual(result["language"], "rust")


if __name__ == "__main__":
    unittest.main()
