import importlib.util
import pathlib
import unittest

MODULE_PATH = pathlib.Path(__file__).with_name("main.py")
SPEC = importlib.util.spec_from_file_location("council_reaction_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CouncilReactionGateTests(unittest.TestCase):
    def test_requires_two_distinct_approvals(self):
        self.assertEqual(MODULE.council_decision({"U1"}, set()), "waiting")
        self.assertEqual(MODULE.council_decision({"U1", "U2"}, set()), "approved")

    def test_blocking_reaction_wins_over_approvals(self):
        self.assertEqual(
            MODULE.council_decision({"U1", "U2", "U3"}, {"U4"}),
            "blocked",
        )

    def test_extracts_reactors_by_emoji_name(self):
        message = {
            "reactions": [
                {"name": "white_check_mark", "users": ["U1", "U2"]},
                {"name": "no_entry_sign", "users": ["U3"]},
            ]
        }
        self.assertEqual(
            MODULE.reaction_users(message, "white_check_mark"),
            {"U1", "U2"},
        )
        self.assertEqual(
            MODULE.reaction_users(message, "no_entry_sign"),
            {"U3"},
        )

    def test_uses_latest_approval_message_marker(self):
        comments = [
            {
                "body": (
                    "council-approval-message: CNEW/200.2\n"
                    "council-pr-author-slack-user: UAUTHOR"
                ),
                "createdAt": "2026-06-16T12:00:00Z",
            },
            {"body": "unrelated", "createdAt": "2026-06-15T12:00:00Z"},
            {
                "body": "council-approval-message: COLD/100.1",
                "createdAt": "2026-06-14T12:00:00Z",
            },
        ]
        self.assertEqual(
            MODULE.parse_approval_message(comments),
            ("CNEW", "200.2", "UAUTHOR"),
        )

    def test_prefers_team_specific_approved_label(self):
        issue = {"team": {"id": "TEAM-1", "name": "Enterprise"}}
        labels = [
            {"id": "WORKSPACE", "team": None},
            {"id": "TEAM", "team": {"id": "TEAM-1", "name": "Enterprise"}},
        ]
        self.assertEqual(MODULE.approved_label_for_issue(labels, issue), "TEAM")

    def test_uses_workspace_label_when_team_label_is_absent(self):
        issue = {"team": {"id": "TEAM-1", "name": "Enterprise"}}
        labels = [{"id": "WORKSPACE", "team": None}]
        self.assertEqual(
            MODULE.approved_label_for_issue(labels, issue),
            "WORKSPACE",
        )

    def test_replies_to_parent_thread_for_approval_reply(self):
        self.assertEqual(
            MODULE.reply_thread_timestamp({"thread_ts": "100.1"}, "200.2"),
            "100.1",
        )
        self.assertEqual(MODULE.reply_thread_timestamp({}, "200.2"), "200.2")

    def test_app_user_is_not_human(self):
        cache = {
            "UAPP": {
                "id": "UAPP",
                "deleted": False,
                "is_bot": False,
                "is_app_user": True,
            }
        }
        self.assertFalse(MODULE.is_human("unused", "UAPP", cache))

    def test_excludes_pr_author_and_nonmember_from_votes(self):
        human = {"deleted": False, "is_bot": False, "is_app_user": False}
        cache = {
            "UAUTHOR": {"id": "UAUTHOR", **human},
            "UCOUNCIL": {"id": "UCOUNCIL", **human},
            "UOUTSIDE": {"id": "UOUTSIDE", **human},
        }
        self.assertEqual(
            MODULE.eligible_reactors(
                "unused",
                {"UAUTHOR", "UCOUNCIL", "UOUTSIDE"},
                {"UAUTHOR", "UCOUNCIL"},
                "UAUTHOR",
                cache,
            ),
            {"UCOUNCIL"},
        )

if __name__ == "__main__":
    unittest.main()
