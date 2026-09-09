"""CLI: python3 tools/cli_meetily.py import transcript <file>."""

from __future__ import annotations

import argparse
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from kanban import KanbanStore, default_db_path  # noqa: E402
from meetily import MeetilyService  # noqa: E402


def import_transcript(
    path: str,
    *,
    store: KanbanStore | None = None,
    board_id: str | None = None,
    board_name: str = "Meetings",
) -> dict:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    kanban = store or KanbanStore(default_db_path())
    service = MeetilyService(kanban)
    if not board_id:
        boards = kanban.list_boards()
        if boards:
            board_id = str(boards[0]["id"])
        else:
            board_id = str(kanban.create_board(board_name)["id"])
    return service.ingest(text, board_id=board_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openhands")
    sub = parser.add_subparsers(dest="cmd")
    import_parser = sub.add_parser("import")
    import_sub = import_parser.add_subparsers(dest="import_cmd")
    transcript = import_sub.add_parser("transcript")
    transcript.add_argument("file")
    transcript.add_argument("--board-id")
    args = parser.parse_args(argv)
    if args.cmd != "import" or args.import_cmd != "transcript":
        parser.print_help()
        return 2
    result = import_transcript(args.file, board_id=args.board_id)
    print(json.dumps({"created": len(result["created"]), "duplicates": len(result["duplicates"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
