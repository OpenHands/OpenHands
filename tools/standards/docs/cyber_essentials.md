# Cyber Essentials plugin — scope and limits

This plugin is a **heuristic checker**. It does **not** issue a Cyber
Essentials (or Cyber Essentials Plus) certificate.

## What it looks for

| Rule | Intent |
|------|--------|
| `CE-DEFAULT-CREDS` | `admin`/`admin`, `password=password`, similar defaults |
| `CE-WEAK-CRYPTO` | MD5 / SHA-1 / DES in hash or cipher calls |
| `CE-PERMISSIVE-CORS` | `Access-Control-Allow-Origin: *` (and close variants) |
| `CE-DEBUG-ROUTE` | `/debug` routes, `--debug`, `debug=True` in `app.run` |
| `CE-DEBUG-FLAG` | `DEBUG = True` (auto-fixable to `False`) |
| `CE-UNBOUNDED-AUTH` | Framework routes with no auth decorator/middleware heuristic |

## False positives

- MD5 used for non-security checksums.
- Public CORS on a truly public asset host.
- Health-check routes that should be unauthenticated.
- `DEBUG = True` in test fixtures (will be flagged).

## What this does NOT guarantee

- Firewall, patching, malware, or access-control questionnaires.
- Coverage of every framework’s auth model.
- That a product “meets Cyber Essentials.” Treat hits as review prompts only.
