from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_COUNTER = itertools.count()


def load_module(script_name: str):
    path = ROOT / 'scripts' / script_name
    module_name = f'test_{path.stem}_{next(MODULE_COUNTER)}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f'Unable to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_agent_message(text: str) -> dict:
    return {
        'kind': 'MessageEvent',
        'source': 'agent',
        'llm_message': {'content': [{'type': 'text', 'text': text}]},
    }


def test_extract_label_names_normalizes_issue_labels():
    module = load_module('issue_good_first_issue_check_openhands.py')

    label_names = module.extract_label_names(
        {
            'labels': [
                {'name': 'bug'},
                'good first issue',
                {'name': 'Bug'},
                {'name': ''},
                None,
            ]
        }
    )

    assert label_names == ['Bug', 'bug', 'good first issue']


def test_build_prompt_includes_issue_labels_and_criteria():
    module = load_module('issue_good_first_issue_check_openhands.py')

    prompt = module.build_prompt(
        'OpenHands/OpenHands',
        {
            'number': 123,
            'title': 'Clarify triage behavior',
            'body': 'The issue needs a clear automated path.',
            'html_url': 'https://github.com/OpenHands/OpenHands/issues/123',
            'labels': [{'name': 'bug'}, {'name': 'frontend'}],
        },
    )

    assert 'Issue labels (JSON array): ["bug", "frontend"]' in prompt
    assert 'Do NOT treat “not a duplicate” as positive evidence on its own.' in prompt
    assert 'All of the following must be true to apply `good first issue`:' in prompt


def test_normalize_result_requires_high_confidence_and_no_disqualifiers():
    module = load_module('issue_good_first_issue_check_openhands.py')

    normalized = module.normalize_result(
        {
            'should_apply_label': True,
            'confidence': 'medium',
            'summary': 'Looks promising',
            'criteria_met': ['narrow scope', 'clear request'],
            'disqualifiers': [],
        }
    )

    assert normalized['should_apply_label'] is False
    assert normalized['confidence'] == 'medium'

    normalized = module.normalize_result(
        {
            'should_apply_label': True,
            'confidence': 'high',
            'summary': 'Looks promising',
            'criteria_met': ['narrow scope', 'clear request'],
            'disqualifiers': ['needs enterprise context'],
        }
    )

    assert normalized['should_apply_label'] is False
    assert normalized['disqualifiers'] == ['needs enterprise context']


def test_normalize_result_limits_string_lists():
    module = load_module('issue_good_first_issue_check_openhands.py')

    normalized = module.normalize_result(
        {
            'should_apply_label': True,
            'confidence': 'high',
            'summary': 'Looks promising',
            'criteria_met': ['1', '2', '3', '4', '5', '6'],
            'disqualifiers': [None, '', 'too broad', 42, ''],
        }
    )

    assert normalized['criteria_met'] == ['1', '2', '3', '4', '5']
    assert normalized['disqualifiers'] == ['too broad', '42']
    assert normalized['should_apply_label'] is False


def test_good_first_issue_main_writes_result_from_final_response(monkeypatch, tmp_path):
    module = load_module('issue_good_first_issue_check_openhands.py')
    output_path = tmp_path / 'result.json'

    monkeypatch.setattr(
        module,
        'parse_args',
        lambda: argparse.Namespace(
            repository='OpenHands/OpenHands',
            issue_number=123,
            output=str(output_path),
            poll_interval_seconds=1,
            max_wait_seconds=10,
        ),
    )
    monkeypatch.setattr(
        module,
        'fetch_issue',
        lambda repository, issue_number: {
            'number': issue_number,
            'title': 'Issue title',
            'body': 'Issue body',
            'html_url': f'https://github.com/{repository}/issues/{issue_number}',
            'labels': [{'name': 'bug'}],
        },
    )
    monkeypatch.setattr(
        module,
        'start_conversation',
        lambda *args, **kwargs: {'app_conversation_id': 'conv-123'},
    )
    monkeypatch.setattr(
        module,
        'poll_conversation',
        lambda app_conversation_id, poll_interval_seconds, max_wait_seconds: {
            'conversation_url': 'https://runtime.example/api/conversations/conv-123',
            'session_api_key': 'session-key',
        },
    )
    monkeypatch.setattr(
        module,
        'fetch_agent_server_final_response',
        lambda app_conversation_id, agent_server_url, session_api_key: json.dumps(
            {
                'should_apply_label': True,
                'confidence': 'high',
                'summary': 'Narrow, clear, and easy to validate.',
                'criteria_met': ['narrow scope', 'clear expected outcome'],
                'disqualifiers': [],
            }
        ),
    )
    monkeypatch.setattr(
        module,
        'fetch_app_server_events',
        lambda app_conversation_id: pytest.fail(
            'fetch_app_server_events should not run'
        ),
    )
    monkeypatch.setattr(
        module,
        'fetch_agent_server_events',
        lambda *args, **kwargs: pytest.fail('fetch_agent_server_events should not run'),
    )

    assert module.main() == 0

    result = json.loads(output_path.read_text())
    assert result['should_apply_label'] is True
    assert result['confidence'] == 'high'
    assert result['criteria_met'] == ['narrow scope', 'clear expected outcome']
    assert result['repository'] == 'OpenHands/OpenHands'


def test_good_first_issue_main_falls_back_to_app_server_events(monkeypatch, tmp_path):
    module = load_module('issue_good_first_issue_check_openhands.py')
    output_path = tmp_path / 'result.json'

    monkeypatch.setattr(
        module,
        'parse_args',
        lambda: argparse.Namespace(
            repository='OpenHands/OpenHands',
            issue_number=123,
            output=str(output_path),
            poll_interval_seconds=1,
            max_wait_seconds=10,
        ),
    )
    monkeypatch.setattr(
        module,
        'fetch_issue',
        lambda repository, issue_number: {
            'number': issue_number,
            'title': 'Issue title',
            'body': 'Issue body',
            'html_url': f'https://github.com/{repository}/issues/{issue_number}',
            'labels': [{'name': 'enhancement'}],
        },
    )
    monkeypatch.setattr(
        module,
        'start_conversation',
        lambda *args, **kwargs: {'app_conversation_id': 'conv-123'},
    )
    monkeypatch.setattr(
        module,
        'poll_conversation',
        lambda app_conversation_id, poll_interval_seconds, max_wait_seconds: {
            'conversation_url': 'https://runtime.example/api/conversations/conv-123',
            'session_api_key': 'session-key',
        },
    )
    monkeypatch.setattr(
        module,
        'fetch_agent_server_final_response',
        lambda *args, **kwargs: '',
    )
    monkeypatch.setattr(
        module,
        'fetch_app_server_events',
        lambda app_conversation_id: [
            make_agent_message(
                json.dumps(
                    {
                        'should_apply_label': False,
                        'confidence': 'medium',
                        'summary': 'Needs more discovery before a newcomer can pick it up.',
                        'criteria_met': ['clear user-facing outcome'],
                        'disqualifiers': ['scope is still too broad'],
                    }
                )
            )
        ],
    )
    monkeypatch.setattr(
        module,
        'fetch_agent_server_events',
        lambda *args, **kwargs: pytest.fail('fetch_agent_server_events should not run'),
    )

    assert module.main() == 0

    result = json.loads(output_path.read_text())
    assert result['should_apply_label'] is False
    assert result['disqualifiers'] == ['scope is still too broad']


def test_good_first_issue_main_rejects_pull_requests(monkeypatch, tmp_path):
    module = load_module('issue_good_first_issue_check_openhands.py')

    monkeypatch.setattr(
        module,
        'parse_args',
        lambda: argparse.Namespace(
            repository='OpenHands/OpenHands',
            issue_number=123,
            output=str(tmp_path / 'result.json'),
            poll_interval_seconds=1,
            max_wait_seconds=10,
        ),
    )
    monkeypatch.setattr(
        module,
        'fetch_issue',
        lambda repository, issue_number: {
            'number': issue_number,
            'pull_request': {'url': 'https://example.test/pr/123'},
        },
    )

    with pytest.raises(RuntimeError, match='is a pull request, not an issue'):
        module.main()
