# UTF-8 tar member names

Mock-mode captures for issue #17704 / PR #19487.

- `before.png` — `origin/main` reader: `café.py` renders as `cafÃ©.py`, with no Entrypoint badge.
- `after.png` — branch reader: `café.py` renders correctly, is badged `入口点`, and sorts first.

The temporary mock fixture used for the captures serves a `packTarGzip` bundle
containing `café.py` and `main.py` from `/api/automation/v1/<id>/tarball`; it is
not part of the PR diff.
