# html-craft - how to build the page

Shared craft for every show-me dimension. One self-contained, offline, editorial HTML
page with hand-drawn SVG figures and code-grounded claims.

## The look (editorial document, light theme)

Calm, readable, document-like - not a dark dashboard. Skeleton:

```html
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{Title}}</title>
<style>
  :root{
    /* single source of truth - prose AND inline-SVG fills both read these (see SVG section) */
    --fg:#1a1a1a; --muted:#5a5a5a; --subtle:#8a8a82;      /* 3 text weights: title · detail · sublabel */
    --bg:#fdfdfb; --card:#ffffff; --elevated:#f7f6f1;     /* page · node fill · raised band/pill */
    --accent:#2b4eaa; --accent-soft:#e6ecfa;
    --rule:#e2e2dc; --rule-strong:#bdbcb4;                /* hairline · emphasized border */
    --code-bg:#f3f1ec; --warn:#b25b0e; --ok:#2f7a3a; --crit:#a02020;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans)}
  body{line-height:1.6;font-size:16px}
  .wrap{display:grid;grid-template-columns:240px minmax(0,1fr);gap:48px;max-width:1180px;margin:0 auto;padding:32px 28px 96px}
  nav.toc{position:sticky;top:24px;align-self:start;font-size:13px}
  nav.toc h2{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
  nav.toc ol{list-style:none;padding:0;margin:0;counter-reset:toc}
  nav.toc li{counter-increment:toc;margin:4px 0}
  nav.toc li::before{content:counter(toc) ". ";color:var(--muted)}
  nav.toc a{color:var(--fg);text-decoration:none} nav.toc a:hover{color:var(--accent)}
  header.title{border-bottom:1px solid var(--rule);padding-bottom:22px;margin-bottom:28px}
  header.title .eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:700}
  header.title h1{font-size:34px;margin:6px 0 4px;letter-spacing:-.01em}
  header.title .sub{color:var(--muted);max-width:740px}
  h2{font-size:22px;margin:44px 0 10px} h2 .num{color:var(--muted);font-weight:500;margin-right:8px}
  h3{font-size:16px;margin:22px 0 8px}
  code{font-family:var(--mono);font-size:.88em;background:var(--code-bg);padding:1px 5px;border-radius:3px}
  pre{font-family:var(--mono);font-size:13px;background:var(--code-bg);padding:14px 16px;border-radius:6px;overflow-x:auto;border:1px solid var(--rule)}
  pre code{background:transparent;padding:0}
  .kw{color:#7048a8}.str{color:var(--ok)}.com{color:var(--subtle);font-style:italic}
  table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
  th{font-weight:600;background:var(--code-bg)}
  .callout{border-left:3px solid var(--accent);background:var(--accent-soft);padding:10px 14px;margin:14px 0;border-radius:0 4px 4px 0;font-size:14.5px}
  .callout.warn{border-color:var(--warn);background:#fbf1e6} .callout.note{border-color:var(--muted);background:#f3f1ec}
  .fig{margin:18px 0 22px}
  .fig svg{display:block;max-width:100%;height:auto;background:#fff;border:1px solid var(--rule);border-radius:6px}
  .fig figcaption{font-size:13px;color:var(--muted);margin-top:6px;text-align:center}
  a.src{font-family:var(--mono);font-size:12px;color:var(--accent);text-decoration:none;border-bottom:1px dotted var(--accent)}
  details{border:1px solid var(--rule);border-radius:6px;margin:12px 0}
  details>summary{cursor:pointer;padding:8px 12px;color:var(--muted);font-size:13px;list-style:none}
  details>summary::-webkit-details-marker{display:none}
  details>summary::before{content:"▸ "} details[open]>summary::before{content:"▾ "}
  .ba{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
  @media(max-width:760px){.wrap{grid-template-columns:1fr}.ba{grid-template-columns:1fr}}
</style></head><body>
<div class="wrap">
  <nav class="toc"><h2>Contents</h2><ol>
    <li><a href="#sec1">…</a></li>
  </ol></nav>
  <main>
    <header class="title"><div class="eyebrow">{{kind}}</div><h1>{{Title}}</h1>
      <p class="sub">{{one-paragraph mental model - the big picture in 2-3 sentences}}</p></header>
    <section id="sec1"><h2><span class="num">1</span>…</h2> … </section>
  </main>
</div></body></html>
```

Numbered sticky TOC + one-paragraph mental model up top + sectioned body. No JS needed
for this shell.

## First screen: 15-second orientation

The reader's attention is the budget. The first viewport should answer four questions
without requiring a full read:

1. **What is this?** A one-paragraph mental model in the title subtitle.
2. **Why does it matter?** The highest-impact conclusion, risk, or action in a `★`
   callout near the top.
3. **Where should I jump?** A numbered sticky TOC whose labels carry information, not
   just categories.
4. **Why should I trust it?** Nearby source links (`path:line`, test, command, commit,
   or fixture) for the first substantive claim.

Do not open with "I read these files" or a chronological work log. Start with the
reader's decision: what changed, how the system works, what path matters, or where to
look next. Put methods, command output, and raw diffs behind `<details>` unless they
are the point of the report.

## Skimmable structure

Design the page so a busy reader can scan headings, captions, callouts, and tables
before choosing where to dive:

- **Headings make claims.** Prefer "Writes only cross the queue" over "Architecture",
  when the section has a specific finding. Generic labels are acceptable only when the
  title/subtitle already carries the finding.
- **One paragraph, one judgment.** Keep paragraphs short; split when a sentence starts
  proving a different point.
- **Number parallel points.** If you say "three rules" or "two risks", number them so
  the reader can reconcile the claim with the list.
- **Use tables only for stable comparison axes.** A table should reduce cognitive
  work, not force subtle judgments into neat boxes.
- **Make captions do work.** A caption states the takeaway of the figure, not merely
  its type.

## Callouts (give them semantic types)

The `.callout` / `.callout.warn` / `.callout.note` styles aren't interchangeable - assign
each a fixed job and the reader learns to skim by them:

- **`★` key takeaway** (accent) - the one load-bearing sentence of a section. At most one per
  section; it's what the reader should remember if they read nothing else.
- **`ⓘ` note** (`.note`, muted) - a reading hint for a figure, an aside, a "why we did it this
  way" that isn't on the critical path.
- **`⚠` warning** (`.warn`) - a boundary, a risk, a gotcha, a guardrail: trust boundaries,
  "this does NOT do X", "never commit the secret". Reserve it for things that bite.

Don't let callouts become wallpaper - if every paragraph is a colored box, none of them carry
weight. A glyph prefix (`★ ⓘ ⚠`) makes the type legible before the reader parses the text.

## Hand-drawn SVG figures (the diagrams that carry the argument)

Draw bespoke inline `<svg viewBox="0 0 W H">` with a `<defs>` for arrow markers and a
scoped `<style>`. You place every box → before/after stays aligned, legible, and the
delta is unmistakable. Semantic class system:

```html
<svg viewBox="0 0 960 460" role="img" aria-labelledby="f1-t f1-d">
 <title id="f1-t">moduleA call path - before vs after</title>
 <desc id="f1-d">Old direct call is replaced by a routed path through the new resolver.</desc>
 <defs>
  <!-- one marker per semantic color; marker fills read the page palette -->
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 Z" fill="var(--subtle)"/></marker>
  <marker id="arr-chg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 Z" fill="var(--warn)"/></marker>
  <!-- subtle "elevated" wash for the box that is the figure's subject -->
  <linearGradient id="subj" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="var(--accent)" stop-opacity=".10"/>
    <stop offset="1" stop-color="var(--accent)" stop-opacity=".02"/></linearGradient>
  <style>
    /* every fill/stroke is a --token: the palette lives once, in :root - prose & figures can't drift */
    .existing{fill:var(--code-bg);stroke:var(--subtle);stroke-width:1.4;stroke-dasharray:5 3} /* before */
    .ex-title{font:700 13px var(--sans);fill:var(--muted)} .ex-detail{font:11.5px var(--mono);fill:var(--subtle)}
    .new{fill:var(--accent-soft);stroke:var(--accent);stroke-width:2}                         /* after / added */
    .new-title{font:800 13px var(--sans);fill:var(--accent)} .detail{font:11.5px var(--mono);fill:var(--fg)}
    .edge{stroke:var(--subtle);stroke-width:1.5;stroke-dasharray:5 3;fill:none}
    .edge-chg{stroke:var(--warn);stroke-width:1.8;fill:none;stroke-dasharray:5 3}             /* the delta */
    .lbl{font:600 10.5px var(--mono);fill:var(--accent)} .gap{font:700 11.5px var(--mono);fill:var(--warn)}
  </style>
 </defs>
 <!-- ===== BEFORE: direct call ===== -->
 <g>
  <rect class="existing" x="40" y="60" width="180" height="74" rx="6"/>
  <text class="ex-title"  x="130" y="90"  text-anchor="middle">moduleA.fn()</text>
  <text class="ex-detail" x="130" y="110" text-anchor="middle">old behavior · a.py:42</text>
 </g>
 <!-- ===== AFTER: routed call (the change) ===== -->
 <g>
  <rect class="new" x="300" y="60" width="180" height="74" rx="6"/>
  <text class="new-title" x="390" y="90" text-anchor="middle">moduleA.fn()</text>
  <path class="edge-chg" d="M220,97 L300,97" marker-end="url(#arr-chg)"/>
  <!-- pill behind an on-edge label so it stays legible over the line -->
  <rect x="236" y="79" width="48" height="16" rx="5" fill="var(--elevated)" stroke="var(--rule)"/>
  <text class="gap" x="260" y="91" text-anchor="middle">new path</text>
 </g>
</svg>
```

**Five craft moves in that skeleton - they're what make a figure read like a designed
diagram instead of a sketch:**
- **Drive every fill/stroke from `var(--token)`.** Define the palette once in `:root`; the
  SVG references it. One source of truth - prose and figures stay in lockstep, and a palette
  tweak reflows every diagram. Never paste a raw hex into a figure.
- **Three text weights, always the same three.** `--fg` for the node title (the real symbol),
  `--muted` for the detail line, `--subtle` for sublabels/`path:line`. The eye sorts the
  hierarchy without reading. Mixing weights ad-hoc is what makes a diagram look noisy.
- **`<title>` + `<desc>` with `aria-labelledby`.** Not decoration: it's the figure's own
  caption-of-record (screen readers, and a note-to-self of what the figure claims).
- **A gradient wash marks the subject.** The one container the figure is *about* gets the
  `#subj` wash (accent 10%→2%); everything else is flat. The reader's eye lands on the
  protagonist before reading a word.
- **`<g>` groups + `<!-- section -->` comments + on-edge label pills.** Author the SVG like
  code: one `<g>` per logical unit, a comment naming it, and a small `rect` pill behind any
  label that sits on a line (a bare label over an edge turns to mush). It stays editable when
  you revise the layout three times.

**The before/after encoding (use consistently):**
- **dashed gray** (`.existing`, `.edge` muted) = what was there before / unchanged context.
- **solid accent** (`.new`) = what this change adds / the after-state.
- **orange dashed** (`.edge-chg`, `.gap`) = the *delta* - the new/changed flow, the gap being filled. This is what the eye should land on.

Node boxes carry a **title (real symbol) + a detail line (behavior + `file:line`)**, so the
diagram is already half-grounded to code.

Two ways to show before/after:
1. **Side-by-side figures** in a `.ba` grid - `<figure>` "Before" | `<figure>` "After". Best when layouts differ a lot.
2. **One overlaid figure** - existing nodes dashed-gray, new nodes/edges solid-accent + orange delta, in the same coordinate space. Best when the change is *additive* to an existing structure (most legible "what's new" read).

## Caption every figure, and hint how to read the hard ones

A figure with no caption makes the reader reverse-engineer your intent. Two cheap habits
fix it:

- **The `<figcaption>` states the takeaway, not a label.** Not "Architecture diagram" -
  *"Requests never touch the DB directly; every write goes through the queue."* The caption
  is the one sentence you'd say pointing at the figure. Number them (`Fig 3 ·`) and
  **cross-reference between figures** (`state machine in Fig 2`, `evolve timing in Fig 5`) so
  a multi-figure report reads as one system at different altitudes, not five loose pictures.
- **For a figure with a non-obvious convention, add a one-line reading hint** in a `.callout.note`
  right under it, naming the single thing that unlocks it: *"Every arrow enters or leaves the
  bus - there are no agent-to-agent arrows, because the protocol is 'read/write the bus'."*
  One sentence that says *how to look* saves the reader a minute of squinting. Put the hint in
  the figure too when you can - an in-diagram `→ §3.2 / Fig 2` link (accent mono) turns the
  picture into a table of contents.

## Ground every claim to code (clickable)

A picture the human can't verify is a liability. Make claims droppable to source:

- Resolve both repository web bases explicitly. For a fork PR, before-state evidence belongs
  to the base repository and merge-base SHA; after-state evidence belongs to the head
  repository and verified head SHA. Query each with
  `gh repo view <owner>/<repo> --json url`, then build links as
  `https://github.com/<o>/<r>/blob/<sha>/<path>#L<n>`. A deleted file must link to the base
  repository at the merge-base SHA; a newly added file has no before-state link.
- Render locations as `<a class="src" href="{{blobURL}}">path:line</a>` in node detail lines, section text, and a per-component "source" link.
- Rule: if a box can't be tied to a symbol+location, it's a *concept* box - style it differently and say so; don't fake a link.
- Put evidence beside the claim it supports. The reader should not have to scroll to a
  bibliography to verify a behavior statement, edge, risk, metric, or test result.
- Commands count as evidence when behavior must be observed: show the command and the
  relevant result near the claim, with full logs collapsed if noisy.

### Grounding discipline (borrowed from DeepWiki)

DeepWiki-style wikis enforce grounding hard - worth copying:

- **Source manifest per section.** Open each major section / component card with a small
  collapsed list of the exact files it's built from:
  `<details><summary>Sources</summary> path/a.py · path/b.py …</details>`. The reader sees
  what the claim rests on before trusting it.
- **Cite after every substantive claim, diagram, table, and snippet.** Inline format:
  `Sources: [path:start-end]` (range) or `[path:line]` (single), multiple allowed. A
  section with no citations is a smell - either ground it or cut it.
- **Solely from the code. Do not invent.** Every statement, box, edge, and number must be
  derived from files you actually read - not from how "similar systems usually work." If
  something important isn't in the code, say so explicitly rather than guessing. Mark genuine
  inferences as inferences.
- **Breadth check.** Decide if necessary, depending on PR. A "comprehensive" page that cites only 1-2 files is probably shallow -
  pull in the related files (callers, callees, config, tests) until the picture is real.

## Code excerpts (when shown)

Keep them short and hand-highlight with spans (no JS highlighter): wrap keywords
`<span class="kw">`, strings `<span class="str">`, comments `<span class="com">`.
**Escape for the insertion context before wrapping** - source text is untrusted. In text
nodes, escape `&`, `<`, and `>`. In quoted HTML or SVG attributes, additionally escape `"`
as `&quot;` and `'` as `&#39;`; always quote attributes. For source URLs, percent-encode the
dynamic owner, repository, and ref as URL path segments; encode each source-path segment
separately and rejoin with `/`. Then attribute-escape the completed URL. Put long diffs/code
inside `<details>` (collapsed). Escape every source-derived value
inserted into HTML or SVG, including titles, symbol names, paths, labels, link text, and
attribute values. Never treat source-derived text as markup.

## Self-contained

- One `.html` file. Inline all CSS and SVG. No build, no framework.
- No external scripts, styles, fonts, images, or other assets. Besides breaking offline use,
  third-party active content can read and disclose a private design document opened locally.

## Mermaid (quick / auxiliary only)

For drafting a throwaway or simple auxiliary graph, locally installed Mermaid can be faster
than hand-SVG. Render it locally to static SVG, inspect and sanitize the output, then inline
that SVG. Never ship a Mermaid runtime or CDN script in the page. For the **carrying**
before/after diagram, hand-SVG wins - auto-layout drifts and breaks the side-by-side alignment
that makes the delta readable.

### Public repositories: commit under `.pr/` and use an htmlpreview link

Because the page is **self-contained**, GitHub can render it through `htmlpreview.github.io`
(it fetches the raw HTML and serves it with the right content-type - plain `raw.githubusercontent.com`
won't, it returns `text/plain`). Commit the file to the PR branch under the temporary `.pr/`
directory, then build the link and paste it in the PR description:

```bash
# commit the html under .pr/ on your PR branch (this dir is temporary, see the skill)
git add .pr/design.html && git commit -m "docs(.pr): design doc" && git push <your-fork> <branch>
# link template - use YOUR fork + PR branch, so it renders before the PR is merged:
#   https://htmlpreview.github.io/?https://github.com/<fork-owner>/<repo>/blob/<pr-branch>/.pr/design.html
```

Example shape:
`https://htmlpreview.github.io/?https://github.com/FORK_OWNER/REPO/blob/PR_BRANCH/.pr/design.html`

`<pr-branch>` can be any branch or commit - `raw.githubusercontent.com` serves it regardless
of merge state, so the doc renders while the PR is still open. Point the URL at the **fork
and branch the PR is opened from**, not `main`.

For public repositories, anyone can open the rendered page without a download or local server.
This works with self-contained pages containing only inline CSS and SVG.

### Private repositories: keep the preview private

`htmlpreview` cannot fetch private repository content. Do not work around that by publishing
the design doc to GitHub Pages or another public host. Keep the document free of external
scripts and assets: opening a local file does not make third-party active content private.

- Link authorized reviewers to the committed GitHub blob and ask them to download and open the
  self-contained HTML file locally; or use an existing access-controlled artifact service.
- For a local browser preview, open `.pr/design.html` directly. If the browser needs HTTP,
  serve only on loopback:
  ```bash
  python -m http.server 8000 --bind 127.0.0.1 --directory .pr
  # open http://127.0.0.1:8000/design.html
  ```
- Use GitHub Pages only when the user explicitly authorizes publication and an administrator
  confirms that private Pages access control is enabled for this repository.
