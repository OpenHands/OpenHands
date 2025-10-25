#!/usr/bin/env python3
"""
Simple main entry point for OpenHands CLI.
This is a simplified version that demonstrates the TUI functionality.
"""

import logging
import os
import sys
import warnings

debug_env = os.getenv('DEBUG', 'false').lower()
if debug_env != '1' and debug_env != 'true':
    logging.disable(logging.WARNING)
    warnings.filterwarnings('ignore')

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import HTML

from openhands_cli.argparsers.main_parser import create_main_parser


def _build_initial_user_message(args) -> str | None:
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                return args.task
            return content
        except Exception:
            return args.task
    return args.task


def main() -> None:
    """Main entry point for the OpenHands CLI.

    Raises:
        ImportError: If agent chat dependencies are missing
        Exception: On other error conditions
    """
    parser = create_main_parser()
    args = parser.parse_args()

    try:
        if args.command == 'serve':
            # Import gui_launcher only when needed
            from openhands_cli.gui_launcher import launch_gui_server

            launch_gui_server(mount_cwd=args.mount_cwd, gpu=args.gpu)
        else:
            # Default CLI behavior - no subcommand needed
            # Import agent_chat only when needed
            from openhands_cli.agent_chat import run_cli_entry

            initial_user_message = _build_initial_user_message(args)
            # Start agent chat
            kwargs = {"resume_conversation_id": args.resume}
            if initial_user_message:
                kwargs["initial_user_message"] = initial_user_message
            run_cli_entry(**kwargs)
    except KeyboardInterrupt:
        print_formatted_text(HTML('\n<yellow>Goodbye! 👋</yellow>'))
    except EOFError:
        print_formatted_text(HTML('\n<yellow>Goodbye! 👋</yellow>'))
    except Exception as e:
        print_formatted_text(HTML(f'<red>Error: {e}</red>'))
        import traceback

        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
