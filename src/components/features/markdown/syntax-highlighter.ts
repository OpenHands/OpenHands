import { PrismLight } from "react-syntax-highlighter";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import batch from "react-syntax-highlighter/dist/esm/languages/prism/batch";
import c from "react-syntax-highlighter/dist/esm/languages/prism/c";
import clike from "react-syntax-highlighter/dist/esm/languages/prism/clike";
import clojure from "react-syntax-highlighter/dist/esm/languages/prism/clojure";
import cpp from "react-syntax-highlighter/dist/esm/languages/prism/cpp";
import csharp from "react-syntax-highlighter/dist/esm/languages/prism/csharp";
import css from "react-syntax-highlighter/dist/esm/languages/prism/css";
import dart from "react-syntax-highlighter/dist/esm/languages/prism/dart";
import diff from "react-syntax-highlighter/dist/esm/languages/prism/diff";
import docker from "react-syntax-highlighter/dist/esm/languages/prism/docker";
import elixir from "react-syntax-highlighter/dist/esm/languages/prism/elixir";
import erlang from "react-syntax-highlighter/dist/esm/languages/prism/erlang";
import fsharp from "react-syntax-highlighter/dist/esm/languages/prism/fsharp";
import go from "react-syntax-highlighter/dist/esm/languages/prism/go";
import graphql from "react-syntax-highlighter/dist/esm/languages/prism/graphql";
import groovy from "react-syntax-highlighter/dist/esm/languages/prism/groovy";
import haskell from "react-syntax-highlighter/dist/esm/languages/prism/haskell";
import hcl from "react-syntax-highlighter/dist/esm/languages/prism/hcl";
import http from "react-syntax-highlighter/dist/esm/languages/prism/http";
import ini from "react-syntax-highlighter/dist/esm/languages/prism/ini";
import java from "react-syntax-highlighter/dist/esm/languages/prism/java";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import json5 from "react-syntax-highlighter/dist/esm/languages/prism/json5";
import jsx from "react-syntax-highlighter/dist/esm/languages/prism/jsx";
import julia from "react-syntax-highlighter/dist/esm/languages/prism/julia";
import kotlin from "react-syntax-highlighter/dist/esm/languages/prism/kotlin";
import less from "react-syntax-highlighter/dist/esm/languages/prism/less";
import lua from "react-syntax-highlighter/dist/esm/languages/prism/lua";
import makefile from "react-syntax-highlighter/dist/esm/languages/prism/makefile";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import markup from "react-syntax-highlighter/dist/esm/languages/prism/markup";
import matlab from "react-syntax-highlighter/dist/esm/languages/prism/matlab";
import nginx from "react-syntax-highlighter/dist/esm/languages/prism/nginx";
import nix from "react-syntax-highlighter/dist/esm/languages/prism/nix";
import objectivec from "react-syntax-highlighter/dist/esm/languages/prism/objectivec";
import ocaml from "react-syntax-highlighter/dist/esm/languages/prism/ocaml";
import perl from "react-syntax-highlighter/dist/esm/languages/prism/perl";
import php from "react-syntax-highlighter/dist/esm/languages/prism/php";
import powershell from "react-syntax-highlighter/dist/esm/languages/prism/powershell";
import properties from "react-syntax-highlighter/dist/esm/languages/prism/properties";
import protobuf from "react-syntax-highlighter/dist/esm/languages/prism/protobuf";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import r from "react-syntax-highlighter/dist/esm/languages/prism/r";
import regex from "react-syntax-highlighter/dist/esm/languages/prism/regex";
import ruby from "react-syntax-highlighter/dist/esm/languages/prism/ruby";
import rust from "react-syntax-highlighter/dist/esm/languages/prism/rust";
import sass from "react-syntax-highlighter/dist/esm/languages/prism/sass";
import scala from "react-syntax-highlighter/dist/esm/languages/prism/scala";
import scss from "react-syntax-highlighter/dist/esm/languages/prism/scss";
import shellSession from "react-syntax-highlighter/dist/esm/languages/prism/shell-session";
import solidity from "react-syntax-highlighter/dist/esm/languages/prism/solidity";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import swift from "react-syntax-highlighter/dist/esm/languages/prism/swift";
import toml from "react-syntax-highlighter/dist/esm/languages/prism/toml";
import tsx from "react-syntax-highlighter/dist/esm/languages/prism/tsx";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";

// Allowlist of grammars to load up-front. Using `Prism` from
// react-syntax-highlighter would pull in all ~300 prism grammars (including
// curiosities like brainfuck) and bloat the bundle, so we register only the
// most common ones here via `PrismLight`.
//
// `clike` is included because several grammars (javascript, csharp, ...)
// extend it at registration time.
const LANGUAGES: Record<string, unknown> = {
  bash,
  batch,
  c,
  clike,
  clojure,
  cpp,
  csharp,
  css,
  dart,
  diff,
  docker,
  elixir,
  erlang,
  fsharp,
  go,
  graphql,
  groovy,
  haskell,
  hcl,
  http,
  ini,
  java,
  javascript,
  json,
  json5,
  jsx,
  julia,
  kotlin,
  less,
  lua,
  makefile,
  markdown,
  markup,
  matlab,
  nginx,
  nix,
  objectivec,
  ocaml,
  perl,
  php,
  powershell,
  properties,
  protobuf,
  python,
  r,
  regex,
  ruby,
  rust,
  sass,
  scala,
  scss,
  "shell-session": shellSession,
  solidity,
  sql,
  swift,
  toml,
  tsx,
  typescript,
  yaml,
};

// Common aliases that prism's language modules don't already declare. The
// per-module `aliases` array (e.g. `js` for javascript) is wired up below
// automatically; this map is for additions on top of that.
const EXTRA_ALIASES: Record<string, string[]> = {
  cpp: ["c++", "cxx", "cc"],
  elixir: ["ex", "exs"],
  erlang: ["erl"],
  fsharp: ["fs", "fsx"],
  hcl: ["terraform", "tf"],
  haskell: ["hs"],
  julia: ["jl"],
  ocaml: ["ml", "mli"],
  powershell: ["ps", "ps1"],
  rust: ["rs"],
  "shell-session": ["console"],
};

Object.entries(LANGUAGES).forEach(([name, lang]) => {
  PrismLight.registerLanguage(name, lang);
  const declared = (lang as { aliases?: unknown }).aliases;
  if (Array.isArray(declared) && declared.length > 0) {
    PrismLight.alias(name, declared as string[]);
  }
  const extra = EXTRA_ALIASES[name];
  if (extra) {
    PrismLight.alias(name, extra);
  }
});

export { PrismLight as SyntaxHighlighter };
