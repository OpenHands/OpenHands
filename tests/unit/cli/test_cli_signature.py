from openhands_cli.argparsers.main_parser import create_main_parser


def test_cli_signature_parses_without_args():
    parser = create_main_parser()
    args = parser.parse_args([])
    # Defaults
    assert getattr(args, 'command', None) is None
    assert getattr(args, 'resume', None) is None
    # user_skills default is True
    assert args.user_skills is True


def test_cli_signature_help_includes_user_skills_flags(capsys):
    parser = create_main_parser()
    try:
        parser.parse_args(['--help'])  # argparse exits SystemExit
    except SystemExit:
        pass
    out = capsys.readouterr().out
    # Verify flags appear in help text
    assert '--user-skills' in out
    assert '--no-user-skills' in out
