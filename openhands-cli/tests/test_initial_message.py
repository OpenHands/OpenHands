import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openhands_cli.simple_main import _build_initial_user_message, main


def test_build_initial_user_message_task_only(tmp_path):
    args = SimpleNamespace(file=None, task="Do the thing")
    assert _build_initial_user_message(args) == "Do the thing"


def test_build_initial_user_message_file_precedence(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("From file content", encoding="utf-8")
    args = SimpleNamespace(file=str(p), task="Fallback task")
    assert _build_initial_user_message(args) == "From file content"


def test_build_initial_user_message_file_missing_falls_back_to_task(tmp_path):
    args = SimpleNamespace(file=str(tmp_path / "missing.txt"), task="Use task")
    assert _build_initial_user_message(args) == "Use task"


def test_build_initial_user_message_empty_file_falls_back_to_task(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("\n\n\t  ", encoding="utf-8")
    args = SimpleNamespace(file=str(p), task="Use task")
    assert _build_initial_user_message(args) == "Use task"


@pytest.mark.parametrize(
    "argv, expected_kwargs",
    [
        (['openhands', '--task', 'Hello'], {"resume_conversation_id": None, "initial_user_message": "Hello"}),
    ],
)
@patch('openhands_cli.agent_chat.run_cli_entry')
def test_main_passes_initial_message_from_task(mock_run, monkeypatch, argv, expected_kwargs):
    monkeypatch.setattr(sys, "argv", argv, raising=False)
    mock_run.side_effect = KeyboardInterrupt()
    main()
    mock_run.assert_called_once_with(**expected_kwargs)


@patch('openhands_cli.agent_chat.run_cli_entry')
def test_main_passes_initial_message_from_file_precedence(mock_run, monkeypatch, tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("Content A", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        'openhands', '--task', 'Task B', '--file', str(p)
    ], raising=False)
    mock_run.side_effect = KeyboardInterrupt()
    main()
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["resume_conversation_id"] is None
    assert call_kwargs["initial_user_message"] == "Content A"


@patch('openhands_cli.agent_chat.run_cli_entry')
def test_main_passes_task_when_file_missing(mock_run, monkeypatch, tmp_path):
    missing = tmp_path / "missing.txt"
    monkeypatch.setattr(sys, "argv", [
        'openhands', '--task', 'Fallback', '--file', str(missing)
    ], raising=False)
    mock_run.side_effect = KeyboardInterrupt()
    main()
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["initial_user_message"] == "Fallback"


@patch('openhands_cli.agent_chat.run_cli_entry')
def test_main_passes_task_when_file_empty(mock_run, monkeypatch, tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("\n  \t", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        'openhands', '--task', 'TaskText', '--file', str(p)
    ], raising=False)
    mock_run.side_effect = KeyboardInterrupt()
    main()
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["initial_user_message"] == "TaskText"
