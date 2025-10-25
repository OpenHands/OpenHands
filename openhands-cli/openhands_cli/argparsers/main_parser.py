"""Main argument parser for OpenHands CLI."""

import argparse


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with CLI as default and serve as subcommand.
    
    Returns:
        The configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='OpenHands CLI - Terminal User Interface for OpenHands AI Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
By default, OpenHands runs in CLI mode (terminal interface).
Use 'serve' subcommand to launch the GUI server instead.

Examples:
  openhands                           # Start CLI mode
  openhands --resume conversation-id  # Resume a conversation in CLI mode
  openhands serve                     # Launch GUI server
  openhands serve --gpu               # Launch GUI server with GPU support
"""
    )
    
    # CLI arguments at top level (default mode)
    parser.add_argument(
        '--resume',
        type=str,
        help='Conversation ID to resume'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        help='Initial task prompt to send once at startup'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to a file whose content will be sent as the initial message'
    )

    # Only serve as subcommand
    subparsers = parser.add_subparsers(
        dest='command',
        help='Additional commands'
    )
    
    # Add serve subcommand
    serve_parser = subparsers.add_parser(
        'serve',
        help='Launch the OpenHands GUI server using Docker (web interface)'
    )
    serve_parser.add_argument(
        '--mount-cwd',
        action='store_true',
        help='Mount the current working directory in the Docker container'
    )
    serve_parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU support in the Docker container'
    )
    
    return parser