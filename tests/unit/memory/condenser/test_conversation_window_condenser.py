"""Unit tests for ConversationWindowCondenser.

These tests mirror the tests for `_apply_conversation_window` in the AgentController,
but adapted to test the condenser implementation. The ConversationWindowCondenser
copies the functionality of the `_apply_conversation_window` function as closely as possible.

The tests verify that the condenser:
1. Identifies essential initial events (System Message, First User Message, Recall Action/Observation)
2. Keeps roughly half of the non-essential events from recent history
3. Handles dangling observations properly
4. Returns appropriate CondensationAction objects specifying which events to forget
"""

from unittest.mock import patch

import pytest

from openhands.events import EventSource
from openhands.events.action import CmdRunAction, MessageAction, RecallAction
from openhands.events.action.agent import CondensationAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    RecallObservation,
)
from openhands.events.recall_type import RecallType
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.condenser.impl.conversation_window_condenser import (
    ConversationWindowCondenser,
)


# Helper function to create events with sequential IDs and causes
def create_events(event_data):
    events = []
    # Import necessary types here to avoid repeated imports inside the loop
    from openhands.events.action import CmdRunAction, RecallAction
    from openhands.events.observation import CmdOutputObservation, RecallObservation

    for i, data in enumerate(event_data):
        event_type = data['type']
        source = data.get('source', EventSource.AGENT)
        kwargs = {}  # Arguments for the event constructor

        # Determine arguments based on event type
        if event_type == RecallAction:
            kwargs['query'] = data.get('query', '')
            kwargs['recall_type'] = data.get('recall_type', RecallType.KNOWLEDGE)
        elif event_type == RecallObservation:
            kwargs['content'] = data.get('content', '')
            kwargs['recall_type'] = data.get('recall_type', RecallType.KNOWLEDGE)
        elif event_type == CmdRunAction:
            kwargs['command'] = data.get('command', '')
        elif event_type == CmdOutputObservation:
            # Required args for CmdOutputObservation
            kwargs['content'] = data.get('content', '')
            kwargs['command'] = data.get('command', '')
            # Pass command_id via kwargs if present in data
            if 'command_id' in data:
                kwargs['command_id'] = data['command_id']
            # Pass metadata if present
            if 'metadata' in data:
                kwargs['metadata'] = data['metadata']
        else:  # Default for MessageAction, SystemMessageAction, etc.
            kwargs['content'] = data.get('content', '')

        # Instantiate the event
        event = event_type(**kwargs)

        # Assign internal attributes AFTER instantiation
        event._id = i + 1  # Assign sequential IDs starting from 1
        event._source = source
        # Assign _cause using cause_id from data, AFTER event._id is set
        if 'cause_id' in data:
            event._cause = data['cause_id']
        # If command_id was NOT passed via kwargs but cause_id exists,
        # pass cause_id as command_id to __init__ via kwargs for legacy handling
        # This needs to happen *before* instantiation if we want __init__ to handle it
        # Let's adjust the logic slightly:
        if event_type == CmdOutputObservation:
            if 'command_id' not in kwargs and 'cause_id' in data:
                kwargs['command_id'] = data['cause_id']  # Let __init__ handle this
            # Re-instantiate if we added command_id
            if 'command_id' in kwargs and event.command_id != kwargs['command_id']:
                event = event_type(**kwargs)
                event._id = i + 1
                event._source = source

        # Now assign _cause if it exists in data, after potential re-instantiation
        if 'cause_id' in data:
            event._cause = data['cause_id']

        events.append(event)
    return events


@pytest.fixture
def condenser_fixture():
    condenser = ConversationWindowCondenser()
    return condenser


# =============================================
# Test Cases for ConversationWindowCondenser
# =============================================


def test_basic_truncation(condenser_fixture):
    condenser = condenser_fixture

    events = create_events(
        [
            {'type': SystemMessageAction, 'content': 'System Prompt'},  # 1
            {
                'type': MessageAction,
                'content': 'User Task 1',
                'source': EventSource.USER,
            },  # 2
            {'type': RecallAction, 'query': 'User Task 1'},  # 3
            {'type': RecallObservation, 'content': 'Recall result', 'cause_id': 3},  # 4
            {'type': CmdRunAction, 'command': 'ls'},  # 5
            {
                'type': CmdOutputObservation,
                'content': 'file1',
                'command': 'ls',
                'cause_id': 5,
            },  # 6
            {'type': CmdRunAction, 'command': 'pwd'},  # 7
            {
                'type': CmdOutputObservation,
                'content': '/dir',
                'command': 'pwd',
                'cause_id': 7,
            },  # 8
            {'type': CmdRunAction, 'command': 'cat file1'},  # 9
            {
                'type': CmdOutputObservation,
                'content': 'content',
                'command': 'cat file1',
                'cause_id': 9,
            },  # 10
        ]
    )
    view = View(events=events)

    # Calculation (RecallAction now essential):
    # History len = 10
    # Essentials = [sys(1), user(2), recall_act(3), recall_obs(4)] (len=4)
    # Non-essential count = 10 - 4 = 6
    # num_recent_to_keep = max(1, 6 // 2) = 3
    # slice_start_index = 10 - 3 = 7
    # recent_events_slice = history[7:] = [obs2(8), cmd3(9), obs3(10)]
    # Validation: remove leading obs2(8). validated_slice = [cmd3(9), obs3(10)]
    # Final = essentials + validated_slice = [sys(1), user(2), recall_act(3), recall_obs(4), cmd3(9), obs3(10)]
    # Expected IDs: [1, 2, 3, 4, 9, 10]. Length 6.
    # Forgotten IDs: [5, 6, 7, 8]
    condensation = condenser.get_condensation(view)

    assert isinstance(condensation, Condensation)
    assert isinstance(condensation.action, CondensationAction)

    # Check the forgotten event IDs
    forgotten_ids = condensation.action.forgotten
    expected_forgotten = [5, 6, 7, 8]
    assert sorted(forgotten_ids) == expected_forgotten


def test_no_system_message(condenser_fixture):
    condenser = condenser_fixture

    events = create_events(
        [
            {
                'type': MessageAction,
                'content': 'User Task 1',
                'source': EventSource.USER,
            },  # 1
            {'type': RecallAction, 'query': 'User Task 1'},  # 2
            {'type': RecallObservation, 'content': 'Recall result', 'cause_id': 2},  # 3
            {'type': CmdRunAction, 'command': 'ls'},  # 4
            {
                'type': CmdOutputObservation,
                'content': 'file1',
                'command': 'ls',
                'cause_id': 4,
            },  # 5
            {'type': CmdRunAction, 'command': 'pwd'},  # 6
            {
                'type': CmdOutputObservation,
                'content': '/dir',
                'command': 'pwd',
                'cause_id': 6,
            },  # 7
            {'type': CmdRunAction, 'command': 'cat file1'},  # 8
            {
                'type': CmdOutputObservation,
                'content': 'content',
                'command': 'cat file1',
                'cause_id': 8,
            },  # 9
        ]
    )
    view = View(events=events)

    # Calculation (RecallAction now essential):
    # History len = 9
    # Essentials = [user(1), recall_act(2), recall_obs(3)] (len=3)
    # Non-essential count = 9 - 3 = 6
    # num_recent_to_keep = max(1, 6 // 2) = 3
    # slice_start_index = 9 - 3 = 6
    # recent_events_slice = history[6:] = [obs2(7), cmd3(8), obs3(9)]
    # Validation: remove leading obs2(7). validated_slice = [cmd3(8), obs3(9)]
    # Final = essentials + validated_slice = [user(1), recall_act(2), recall_obs(3), cmd3(8), obs3(9)]
    # Expected kept IDs: [1, 2, 3, 8, 9]. Length 5.
    # Forgotten IDs: [4, 5, 6, 7]
    condensation = condenser.get_condensation(view)

    assert isinstance(condensation, Condensation)
    assert isinstance(condensation.action, CondensationAction)

    forgotten_ids = condensation.action.forgotten
    expected_forgotten = [4, 5, 6, 7]
    assert sorted(forgotten_ids) == expected_forgotten


def test_no_recall_observation(condenser_fixture):
    condenser = condenser_fixture

    events = create_events(
        [
            {'type': SystemMessageAction, 'content': 'System Prompt'},  # 1
            {
                'type': MessageAction,
                'content': 'User Task 1',
                'source': EventSource.USER,
            },  # 2
            {'type': RecallAction, 'query': 'User Task 1'},  # 3 (Recall Action exists)
            # Recall Observation is missing
            {'type': CmdRunAction, 'command': 'ls'},  # 4
            {
                'type': CmdOutputObservation,
                'content': 'file1',
                'command': 'ls',
                'cause_id': 4,
            },  # 5
            {'type': CmdRunAction, 'command': 'pwd'},  # 6
            {
                'type': CmdOutputObservation,
                'content': '/dir',
                'command': 'pwd',
                'cause_id': 6,
            },  # 7
            {'type': CmdRunAction, 'command': 'cat file1'},  # 8
            {
                'type': CmdOutputObservation,
                'content': 'content',
                'command': 'cat file1',
                'cause_id': 8,
            },  # 9
        ]
    )
    view = View(events=events)

    # Calculation (RecallAction essential even without RecallObs in condenser):
    # History len = 9
    # Essentials = [sys(1), user(2), recall_action(3)] (len=3)
    # Non-essential count = 9 - 3 = 6
    # num_recent_to_keep = max(1, 6 // 2) = 3
    # slice_start_index = 9 - 3 = 6
    # recent_events_slice = history[6:] = [obs2(7), cmd3(8), obs3(9)]
    # Validation: remove leading obs2(7). validated_slice = [cmd3(8), obs3(9)]
    # Final = essentials + validated_slice = [sys(1), user(2), recall_action(3), cmd_cat(8), obs_cat(9)]
    # Expected kept IDs: [1, 2, 3, 8, 9]. Length 5.
    # Forgotten IDs: [4, 5, 6, 7]
    condensation = condenser.get_condensation(view)

    assert isinstance(condensation, Condensation)
    assert isinstance(condensation.action, CondensationAction)

    forgotten_ids = condensation.action.forgotten
    expected_forgotten = [4, 5, 6, 7]
    assert sorted(forgotten_ids) == expected_forgotten


def test_short_history_no_truncation(condenser_fixture):
    condenser = condenser_fixture

    events = create_events(
        [
            {'type': SystemMessageAction, 'content': 'System Prompt'},  # 1
            {
                'type': MessageAction,
                'content': 'User Task 1',
                'source': EventSource.USER,
            },  # 2
            {'type': RecallAction, 'query': 'User Task 1'},  # 3
            {'type': RecallObservation, 'content': 'Recall result', 'cause_id': 3},  # 4
            {'type': CmdRunAction, 'command': 'ls'},  # 5
            {
                'type': CmdOutputObservation,
                'content': 'file1',
                'command': 'ls',
                'cause_id': 5,
            },  # 6
        ]
    )
    view = View(events=events)

    # Calculation (RecallAction now essential):
    # History len = 6
    # Essentials = [sys(1), user(2), recall_act(3), recall_obs(4)] (len=4)
    # Non-essential count = 6 - 4 = 2
    # num_recent_to_keep = max(1, 2 // 2) = 1
    # slice_start_index = 6 - 1 = 5
    # recent_events_slice = history[5:] = [obs1(6)]
    # Validation: remove leading obs1(6). validated_slice = []
    # Final = essentials + validated_slice = [sys(1), user(2), recall_act(3), recall_obs(4)]
    # Expected kept IDs: [1, 2, 3, 4]. Length 4.
    # Forgotten IDs: [5, 6]
    condensation = condenser.get_condensation(view)

    assert isinstance(condensation, Condensation)
    assert isinstance(condensation.action, CondensationAction)

    forgotten_ids = condensation.action.forgotten
    expected_forgotten = [5, 6]
    assert sorted(forgotten_ids) == expected_forgotten


def test_only_essential_events(condenser_fixture):
    condenser = condenser_fixture

    events = create_events(
        [
            {'type': SystemMessageAction, 'content': 'System Prompt'},  # 1
            {
                'type': MessageAction,
                'content': 'User Task 1',
                'source': EventSource.USER,
            },  # 2
            {'type': RecallAction, 'query': 'User Task 1'},  # 3
            {'type': RecallObservation, 'content': 'Recall result', 'cause_id': 3},  # 4
        ]
    )
    view = View(events=events)

    # Calculation (RecallAction now essential):
    # History len = 4
    # Essentials = [sys(1), user(2), recall_act(3), recall_obs(4)] (len=4)
    # Non-essential count = 4 - 4 = 0
    # num_recent_to_keep = max(1, 0 // 2) = 1
    # slice_start_index = 4 - 1 = 3
    # recent_events_slice = history[3:] = [recall_obs(4)]
    # Validation: remove leading recall_obs(4). validated_slice = []
    # Final = essentials + validated_slice = [sys(1), user(2), recall_act(3), recall_obs(4)]
    # Expected kept IDs: [1, 2, 3, 4]. Length 4.
    # Forgotten IDs: []
    condensation = condenser.get_condensation(view)

    assert isinstance(condensation, Condensation)
    assert isinstance(condensation.action, CondensationAction)

    forgotten_ids = condensation.action.forgotten
    expected_forgotten = []
    assert forgotten_ids == expected_forgotten


def test_dangling_observations_at_cut_point(condenser_fixture):
    condenser = condenser_fixture

    events = create_events(
        [
            {'type': SystemMessageAction, 'content': 'System Prompt'},  # 1
            {
                'type': MessageAction,
                'content': 'User Task 1',
                'source': EventSource.USER,
            },  # 2
            {'type': RecallAction, 'query': 'User Task 1'},  # 3
            {'type': RecallObservation, 'content': 'Recall result', 'cause_id': 3},  # 4
            # --- Slice calculation should start here ---
            {
                'type': CmdOutputObservation,
                'content': 'dangle1',
                'command': 'cmd_unknown',
            },  # 5 (Dangling)
            {
                'type': CmdOutputObservation,
                'content': 'dangle2',
                'command': 'cmd_unknown',
            },  # 6 (Dangling)
            {'type': CmdRunAction, 'command': 'cmd1'},  # 7
            {
                'type': CmdOutputObservation,
                'content': 'obs1',
                'command': 'cmd1',
                'cause_id': 7,
            },  # 8
            {'type': CmdRunAction, 'command': 'cmd2'},  # 9
            {
                'type': CmdOutputObservation,
                'content': 'obs2',
                'command': 'cmd2',
                'cause_id': 9,
            },  # 10
        ]
    )  # 10 events total
    view = View(events=events)

    # Calculation (RecallAction now essential):
    # History len = 10
    # Essentials = [sys(1), user(2), recall_act(3), recall_obs(4)] (len=4)
    # Non-essential count = 10 - 4 = 6
    # num_recent_to_keep = max(1, 6 // 2) = 3
    # slice_start_index = 10 - 3 = 7
    # recent_events_slice = history[7:] = [obs1(8), cmd2(9), obs2(10)]
    # Validation: remove leading obs1(8). validated_slice = [cmd2(9), obs2(10)]
    # Final = essentials + validated_slice = [sys(1), user(2), recall_act(3), recall_obs(4), cmd2(9), obs2(10)]
    # Expected kept IDs: [1, 2, 3, 4, 9, 10]. Length 6.
    # Forgotten IDs: [5, 6, 7, 8]
    condensation = condenser.get_condensation(view)

    assert isinstance(condensation, Condensation)
    assert isinstance(condensation.action, CondensationAction)

    forgotten_ids = condensation.action.forgotten
    expected_forgotten = [5, 6, 7, 8]
    assert sorted(forgotten_ids) == expected_forgotten


def test_only_dangling_observations_in_recent_slice(condenser_fixture):
    condenser = condenser_fixture

    events = create_events(
        [
            {'type': SystemMessageAction, 'content': 'System Prompt'},  # 1
            {
                'type': MessageAction,
                'content': 'User Task 1',
                'source': EventSource.USER,
            },  # 2
            {'type': RecallAction, 'query': 'User Task 1'},  # 3
            {'type': RecallObservation, 'content': 'Recall result', 'cause_id': 3},  # 4
            # --- Slice calculation should start here ---
            {
                'type': CmdOutputObservation,
                'content': 'dangle1',
                'command': 'cmd_unknown',
            },  # 5 (Dangling)
            {
                'type': CmdOutputObservation,
                'content': 'dangle2',
                'command': 'cmd_unknown',
            },  # 6 (Dangling)
        ]
    )  # 6 events total
    view = View(events=events)

    # Calculation (RecallAction now essential):
    # History len = 6
    # Essentials = [sys(1), user(2), recall_act(3), recall_obs(4)] (len=4)
    # Non-essential count = 6 - 4 = 2
    # num_recent_to_keep = max(1, 2 // 2) = 1
    # slice_start_index = 6 - 1 = 5
    # recent_events_slice = history[5:] = [dangle2(6)]
    # Validation: remove leading dangle2(6). validated_slice = [] (Corrected based on user feedback/bugfix)
    # Final = essentials + validated_slice = [sys(1), user(2), recall_act(3), recall_obs(4)]
    # Expected kept IDs: [1, 2, 3, 4]. Length 4.
    # Forgotten IDs: [5, 6]
    with patch(
        'openhands.memory.condenser.impl.conversation_window_condenser.logger.warning'
    ) as mock_log_warning:
        condensation = condenser.get_condensation(view)

        assert isinstance(condensation, Condensation)
        assert isinstance(condensation.action, CondensationAction)

        forgotten_ids = condensation.action.forgotten
        expected_forgotten = [5, 6]
        assert sorted(forgotten_ids) == expected_forgotten

        # Check that the specific warning was logged exactly once
        assert mock_log_warning.call_count == 1

        # Check the essential parts of the arguments
        call_args, call_kwargs = mock_log_warning.call_args
        expected_message_substring = 'All recent events are dangling observations, which we truncate. This means the agent has only the essential first events. This should not happen.'
        assert expected_message_substring in call_args[0]


def test_empty_history(condenser_fixture):
    condenser = condenser_fixture
    view = View(events=[])

    condensation = condenser.get_condensation(view)

    assert isinstance(condensation, Condensation)
    assert isinstance(condensation.action, CondensationAction)
    assert condensation.action.forgotten == []


def test_multiple_user_messages(condenser_fixture):
    condenser = condenser_fixture

    events = create_events(
        [
            {'type': SystemMessageAction, 'content': 'System Prompt'},  # 1
            {
                'type': MessageAction,
                'content': 'User Task 1',
                'source': EventSource.USER,
            },  # 2 (First)
            {'type': RecallAction, 'query': 'User Task 1'},  # 3
            {
                'type': RecallObservation,
                'content': 'Recall result 1',
                'cause_id': 3,
            },  # 4
            {'type': CmdRunAction, 'command': 'cmd1'},  # 5
            {
                'type': CmdOutputObservation,
                'content': 'obs1',
                'command': 'cmd1',
                'cause_id': 5,
            },  # 6
            {
                'type': MessageAction,
                'content': 'User Task 2',
                'source': EventSource.USER,
            },  # 7 (Second)
            {'type': RecallAction, 'query': 'User Task 2'},  # 8
            {
                'type': RecallObservation,
                'content': 'Recall result 2',
                'cause_id': 8,
            },  # 9
            {'type': CmdRunAction, 'command': 'cmd2'},  # 10
            {
                'type': CmdOutputObservation,
                'content': 'obs2',
                'command': 'cmd2',
                'cause_id': 10,
            },  # 11
        ]
    )  # 11 events total
    view = View(events=events)

    # Calculation (RecallAction now essential):
    # History len = 11
    # Essentials = [sys(1), user1(2), recall_act1(3), recall_obs1(4)] (len=4)
    # Non-essential count = 11 - 4 = 7
    # num_recent_to_keep = max(1, 7 // 2) = 3
    # slice_start_index = 11 - 3 = 8
    # recent_events_slice = history[8:] = [recall_obs2(9), cmd2(10), obs2(11)]
    # Validation: remove leading recall_obs2(9). validated_slice = [cmd2(10), obs2(11)]
    # Final = essentials + validated_slice = [sys(1), user1(2), recall_act1(3), recall_obs1(4)] + [cmd2(10), obs2(11)]
    # Expected kept IDs: [1, 2, 3, 4, 10, 11]. Length 6.
    # Forgotten IDs: [5, 6, 7, 8, 9]
    condensation = condenser.get_condensation(view)

    assert isinstance(condensation, Condensation)
    assert isinstance(condensation.action, CondensationAction)

    forgotten_ids = condensation.action.forgotten
    expected_forgotten = [5, 6, 7, 8, 9]
    assert sorted(forgotten_ids) == expected_forgotten

    # Additional validation: ensure that only the first user message is kept
    kept_event_ids = set(range(1, 12)) - set(forgotten_ids)
    assert 2 in kept_event_ids  # First user message kept
    assert 7 not in kept_event_ids  # Second user message forgotten
