# GDPR plugin — scope and limits

This plugin is a **heuristic checker**. It does **not** certify GDPR
compliance, complete a DPIA, or prove lawful basis.

## What it looks for

| Rule | Intent |
|------|--------|
| `GDPR-EMAIL` | Email addresses that are not `example.com` / `localhost` fixtures |
| `GDPR-PHONE` | International-looking phone numbers (`+33`, `+44`, `+1`, …) |
| `GDPR-PERSONAL-ID` | Labelled PESEL / DNI / NINO patterns |
| `GDPR-PII-LOG` | `print` / logger / `console.log` that appears to emit PII |
| `GDPR-CONSENT` | Email collection (`type="email"`) without consent/opt-in wording |
| `GDPR-ERASURE` | Collection in code files without a deletion/erasure marker |
| `GDPR-UNENCRYPTED` | `pickle.dump` / `json.dump` / `sqlite3.connect` near user/PII terms |

## False positives

- Docs, tests, and comments that mention emails or “consent”.
- Fixture data that does not use `example.com`.
- A deletion API defined in another file (this check is per-file).

## What this does NOT guarantee

- Lawful basis, DPIA, records of processing, or cross-border transfer controls.
- Detection of PII in binaries, images, or encrypted blobs.
- That a product is “GDPR compliant.” Treat hits as review prompts only.
