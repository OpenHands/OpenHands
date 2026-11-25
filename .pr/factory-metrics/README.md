# Factory metrics display: design hand-off

Design discussion for the metric cards on a Factory's pages. This folder is temporary PR material
(`.pr/` is removed from `main` by the next release PR). It holds the design sheet, its editable source,
and the reference material needed to revise it.

- **Open the sheet:** `index.html` (self-contained, no script). Reviewers use the htmlpreview link in the PR body.
- **Plugin declarations:** `declarations.html` is the overview. One file for each card type sits in `declarations/` (`metrics.html`, `count.html`, and more to come). Each example shows the card, the plugin manifest that sets it up, what the plugin sends, and what the card reads.
- **Status:** exploratory. The open decision is the range of metrics Factories will support, which narrows the
  card types worth building. Nothing here is built.
- **Not a spec.** The Notion pages and the merged backend are inspiration. Where they disagree, the sheet says so
  (sheets 09, 10 and 12).

## Layout

```
index.html          built output: charts as static SVG, no script
declarations.html   built overview of the plugin declaration pages
declarations/       built card-type pages (metrics.html, count.html)
src/index.html      editable source: 14 sheets, inline <style>, build-time chart script
src/declarations/   editable sources: template.html (shared style, page list, footer) and one fragment for each page
src/build.py        renders the sources to ../
reference/          inputs the design was derived from (see reference/README.md)
```

## Revise the design

1. Edit `src/index.html` or a fragment in `src/declarations/`. The declaration pages share one `<style>` block in `src/declarations/template.html`. Never edit a built file by hand. If you change a shared rule such as `.mc`, change it in `src/index.html` and in the template.
2. Rebuild: `python3 .pr/factory-metrics/src/build.py` (needs `chromium` or `google-chrome`).
3. Look at the result before pushing. Headless screenshot of the built file:
   `chromium --headless=new --no-sandbox --window-size=1600,24000 --virtual-time-budget=3000 --screenshot=out.png file://$PWD/.pr/factory-metrics/index.html`
4. Commit `src/` and the rebuilt `index.html` together, then refresh the htmlpreview link in the PR body
   with the new commit SHA (a SHA link keeps working after the branch is deleted).

### Rules the page follows

- **Self-contained and static.** No external URLs, fonts or scripts in the built file. The `<script data-build="strip">`
  in `src/index.html` only draws the sparklines and bars at build time and is removed by `build.py`.
- **Doc chrome is light, artboards are dark.** The page around the drawings is neutral grey and white. Every card
  drawing uses the Agent Canvas dark tokens (`reference/prototype/tokens.css.txt`). Use the token names in text, not hex values.
- **Pixel values are the spec.** Card sizes, paddings and type sizes in sheets 02 to 05 are written as numbers and
  drawn 1:1. If you change a number, change the drawing and every sheet that repeats it.
- **Sheet 03 is absolutely positioned.** The anatomy redline places every element and pin by pixel coordinates inside a
  340 x 186 card. Moving one element means moving its dashed box, pin and leader line.
- **Chips carry provenance.** `existing` (already in the prototype), `new` (proposed here), `merged` / `open PR` / `gap`
  (state of the backend in `OpenHands/automation`). Keep them accurate when facts change.
- **Add a declaration page:** write `src/declarations/<name>.html` (a cover and sheets, with `<section class="sheet" id="sNN">`), add it to `DECLARATION_PAGES` in `src/build.py`, and list it on the overview. Link to other pages with `{{to:declarations/<name>.html}}` and to the overview with `{{overview}}`.
- **Mark checked answers.** A request and answer marked `checked` must come from running the request against the service code. Seed the facts through `POST /v1/outcomes`, and read with the aggregate route. Do not mark a hand-worked number.
- **Add a card type:** add a row to the index table in sheet 05, an `<article class="spec">` with artboard, spec table and
  JSON, then update sheet 09 (field map), sheet 10 (metric map) and sheet 12 (gaps).
- **Writing style:** short sentences, one idea each, plain words. Say what a field is and where it comes from.

### Checks before pushing

- `grep -c "<script" .pr/factory-metrics/index.html .pr/factory-metrics/declarations.html .pr/factory-metrics/declarations/*.html` prints `0` for every file.
- No `.tsx` or `.css` files under `.pr/`. The PR description check treats them as frontend code and then requires screenshots. Reference copies use a `.txt` suffix, and the styles stay inline in `src/index.html`.
- Backend claims still match `OpenHands/automation` `main`. Re-check the open PRs listed in `reference/automation/PRS.md`.

## Where the numbers come from

| Topic | Source |
| --- | --- |
| Type scale, colors, spacing, radius, hover | `reference/prototype/tokens.css.txt`, `factory-ui.tsx.txt` |
| Existing KPI tile, Goal card, flow row | `reference/prototype/overview-pages.tsx.txt`, `flowstats.tsx.txt` |
| What the backend stores and aggregates today | `reference/automation/*` |
| Metric, point, card model; first-release metric list | Notion pages, digest in `reference/notion/README.md` |
