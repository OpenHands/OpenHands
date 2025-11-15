from openhands_cli.argparsers.main_parser import create_main_parser


def test_user_skills_default_true():
    parser = create_main_parser()
    args = parser.parse_args([])
    assert args.user_skills is True


def test_user_skills_disable_with_flag():
    parser = create_main_parser()
    args = parser.parse_args(['--no-user-skills'])
    assert args.user_skills is False


def test_user_skills_enable_with_flag_overrides_disable_if_both_ordered():
    parser = create_main_parser()
    # argparse takes last occurrence wins for store_true/false of same dest
    args = parser.parse_args(['--no-user-skills', '--user-skills'])
    assert args.user_skills is True


def test_user_skills_disable_then_enable_then_disable_last_wins():
    parser = create_main_parser()
    args = parser.parse_args(['--user-skills', '--no-user-skills'])
    assert args.user_skills is False
