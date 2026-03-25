# Plan: reusable contributor wall for GitHub READMEs

## Goal

Create a separate reusable project that lets **any GitHub repository** show a clean, modern wall of contributor photos in its `README.md`.

The ideal outcome is something as easy to embed as Open Collective's contributor image, but based on **GitHub contributors** instead of Open Collective donors.

A repo should be able to add something like this to its README:

```md
[![Contributors](https://example.com/OpenHands/OpenHands.svg)](https://github.com/OpenHands/OpenHands/graphs/contributors)
```

or, for teams that prefer not to depend on a hosted service:

```yaml
- uses: contributor-wall/action@v1
  with:
    output: .github/assets/contributors.svg
```

and then:

```md
[![Contributors](./.github/assets/contributors.svg)](https://github.com/OWNER/REPO/graphs/contributors)
```

## What we want the finished product to do

- Show a **photo/avatar wall** of contributors from a GitHub repository
- Exclude bots by default
- Look cleaner and more modern than a Markdown table
- Be easy to drop into **any GitHub README**
- Require little or no manual curation
- Link back to the repository's contributor graph
- Work for public repositories without custom frontend code in the README itself

## Key constraint: GitHub README rendering

A GitHub README can render Markdown plus a limited subset of raw HTML, but it **cannot run custom JavaScript** and is not a place where we can rely on custom CSS for a polished interactive layout.

That means the final output embedded into a README must be one of these:

1. Plain Markdown
2. Plain HTML (`<a>`, `<img>`, etc.)
3. A generated image asset such as **SVG** or PNG

### Implication

If we want the result to look **clean, modern, and consistent across repos**, the best display format is a **generated SVG**.

That gives us control over:
- circular avatars
- spacing
- wrapping/grid layout
- labels
- theme variants
- empty/loading states

Without requiring JavaScript in the README.

## Is this possible?

**Yes, with constraints.**

### What is possible

- Fetch contributors from the GitHub API
- Filter out bots
- Render a contributor wall as SVG
- Let any repo embed the SVG in its README
- Support both hosted and self-generated modes

### What is not realistic inside a README alone

- Rich client-side interactivity
- Custom scripts
- A highly dynamic layout powered directly by browser-side API calls from the README

So the project should not try to make the README itself smart. The project should generate a **smart asset** that the README can display.

## Recommended product shape

Build a new repo that provides **two ways to use the same renderer**.

### Option A: hosted image service (best UX for consumers)

Consumer README usage:

```md
[![Contributors](https://contribwall.dev/OpenHands/OpenHands.svg?limit=60)](https://github.com/OpenHands/OpenHands/graphs/contributors)
```

#### Pros
- easiest integration
- single-line README install
- closest to Open Collective's UX
- no workflow setup required in the consumer repo

#### Cons
- requires hosting and maintenance
- API rate limiting and caching must be handled centrally
- service availability becomes a dependency

### Option B: GitHub Action + CLI (best MVP)

Consumer repo installs an action that:
1. fetches contributors from the GitHub API
2. filters them
3. renders an SVG
4. commits or updates the generated asset

Consumer README usage stays simple:

```md
[![Contributors](./.github/assets/contributors.svg)](https://github.com/OWNER/REPO/graphs/contributors)
```

#### Pros
- no central hosting required
- works for any public repo
- predictable and cache-friendly
- easier to ship first

#### Cons
- slightly more setup for consumers
- generated files live in each consumer repo
- updates happen on a schedule or on demand, not on every page load

## Recommendation

### MVP recommendation

Start with **Option B: GitHub Action + CLI that generates SVG**.

Why:
- it is the most realistic way to launch quickly
- it avoids building a production web service first
- it still gives repos a polished README result
- the same rendering code can later power a hosted image endpoint

### Phase 2

After the renderer is solid, add **Option A** as a hosted convenience layer:
- same SVG renderer
- same filtering logic
- HTTP caching on top

## Data source

Use the GitHub REST API endpoint for repository contributors:
- `GET /repos/{owner}/{repo}/contributors`

Primary repo link for humans:
- `https://github.com/{owner}/{repo}/graphs/contributors`

## Filtering rules

Default filtering should exclude:
- entries with `type == "Bot"`
- logins ending with `[bot]`
- an optional custom exclude list supplied by the consumer

Examples:
- `dependabot[bot]`
- `github-actions[bot]`
- repo-specific automation users like `openhands-agent`

### Config options

Consumer-configurable settings should include:
- `exclude`: comma-separated usernames
- `include`: optional allowlist override
- `max_contributors`
- `sort`: by contributions or alphabetical
- `avatar_size`
- `columns`
- `theme`: light/dark/neutral
- `show_names`: true/false
- `show_contribution_counts`: true/false

## Desired visual style

The output should feel more modern than `all-contributors`' default table.

### Design principles
- avatar-first
- generous spacing
- no table borders
- centered layout
- responsive wrapping where possible within the SVG canvas
- optional names below avatars, kept minimal
- aesthetically pleasing at both full width and narrower README widths

### Good defaults
- circular 48px avatars
- 10-14 avatars per row depending on width
- subtle title: `Contributors`
- optional footer link text: `View all contributors on GitHub`

## Why SVG is the right rendering target

SVG is ideal because it:
- renders well in GitHub READMEs
- can be embedded as a normal image
- supports clean layout and styling
- can contain clipped circular avatars
- can be regenerated deterministically
- works both for hosted and checked-in asset workflows

## Suggested repo contents for the new project

```text
contributor-wall/
├── packages/
│   ├── core/               # fetch + filter + render logic
│   ├── cli/                # command line interface
│   └── action/             # GitHub Action wrapper
├── examples/
│   ├── basic-readme/
│   ├── scheduled-update/
│   └── hosted-image-usage/
├── docs/
│   ├── installation.md
│   ├── configuration.md
│   └── theming.md
└── README.md
```

## Proposed implementation plan

### Phase 1: renderer spike
- Fetch contributors from GitHub API
- Apply bot filtering
- Render a basic SVG grid of avatars with optional visible labels
- Validate that the SVG looks good in a GitHub README

### Phase 2: CLI
- Add a CLI such as:
  - `contributor-wall generate --repo OpenHands/OpenHands --output contributors.svg`
- Support config via flags and config file
- Support GitHub token for higher rate limits

### Phase 3: GitHub Action
- Package the CLI as an action
- Add scheduled and manual triggers
- Optionally commit generated `contributors.svg` back to the repo
- Support README marker replacement in addition to standalone asset generation

### Phase 4: hosted mode
- Add a small service that serves:
  - `/{owner}/{repo}.svg`
- Cache API results aggressively
- Respect GitHub rate limits and ETags
- Reuse the exact same renderer from the CLI/action

## README integration modes to support

### Mode 1: image-only embed

**Note:** image-only mode is the best-looking and easiest to embed, but it gives you one overall clickable image, not individually clickable avatars.

```md
[![Contributors](https://service.dev/OWNER/REPO.svg)](https://github.com/OWNER/REPO/graphs/contributors)
```

### Mode 2: local asset embed

```md
[![Contributors](./.github/assets/contributors.svg)](https://github.com/OWNER/REPO/graphs/contributors)
```

### Mode 3: marker-based README injection

```md
<!-- contributor-wall:start -->
<!-- contributor-wall:end -->
```

The action can replace the block with generated HTML or Markdown if the consumer wants inline markup rather than a standalone image.

## Tradeoff summary

### Best ease-of-use for consumers
Hosted SVG endpoint

### Best first version to build
GitHub Action + CLI that generates an SVG file

### Best-looking output inside GitHub README constraints
Generated SVG

### Worst fit
Trying to do a fancy live layout directly in README HTML without generating an image

## Open questions

- Should the project show **all** contributors or cap to a configurable top N?
- Should names be shown by default, or only avatars?
- Should maintainers be able to pin or highlight specific contributors?
- Should contributors be grouped by contribution type later, or only code contributors from GitHub for v1?
- Should the hosted service support private repos, or only public repos?

## Success criteria

The project is successful if a maintainer of any public GitHub repo can:
1. install it in a few minutes
2. add one image/snippet to a README
3. get a polished contributor wall
4. avoid bot avatars by default
5. keep it updated with near-zero manual work

## Bottom line

This is **possible**.

The cleanest cross-repo solution is **not** a custom README table. It is a **generated SVG contributor wall** backed by GitHub contributor data.

If we build the renderer first, then package it as a CLI + GitHub Action, we can ship a practical MVP quickly and still leave room for a hosted one-line embed experience later.

## Reference points

- ORY README contributor image pattern:
  `https://github.com/ory/kratos/blob/master/README.md#many-thanks-to-all-individual-contributors`
- all-contributors project:
  `https://github.com/all-contributors/allcontributors.org`
- GitHub docs note that GitHub writing supports Markdown plus some HTML:
  `https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github`
- GitHub REST API repositories docs (includes contributor endpoints):
  `https://docs.github.com/en/rest/repos/repos#list-repository-contributors`
