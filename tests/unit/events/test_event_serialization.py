from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.action.action import ActionSecurityRisk
from openhands.events.metrics import Cost, Metrics, ResponseLatency, TokenUsage
from openhands.events.observation import CmdOutputMetadata, CmdOutputObservation
from openhands.events.serialization import event_from_dict, event_to_dict


def test_command_output_success_serialization():
    # Test successful command
    obs = CmdOutputObservation(
        command='ls',
        content='file1.txt\nfile2.txt',
        metadata=CmdOutputMetadata(exit_code=0),
    )
    serialized = event_to_dict(obs)
    assert serialized['success'] is True

    # Test failed command
    obs = CmdOutputObservation(
        command='ls',
        content='No such file or directory',
        metadata=CmdOutputMetadata(exit_code=1),
    )
    serialized = event_to_dict(obs)
    assert serialized['success'] is False


def test_metrics_basic_serialization():
    # Create a basic action with only accumulated_cost
    action = MessageAction(content='Hello, world!')
    metrics = Metrics()
    metrics.accumulated_cost = 0.03
    action._llm_metrics = metrics

    # Test serialization
    serialized = event_to_dict(action)
    assert 'llm_metrics' in serialized
    assert serialized['llm_metrics']['accumulated_cost'] == 0.03
    assert serialized['llm_metrics']['costs'] == []
    assert serialized['llm_metrics']['response_latencies'] == []
    assert serialized['llm_metrics']['token_usages'] == []

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.llm_metrics is not None
    assert deserialized.llm_metrics.accumulated_cost == 0.03
    assert len(deserialized.llm_metrics.costs) == 0
    assert len(deserialized.llm_metrics.response_latencies) == 0
    assert len(deserialized.llm_metrics.token_usages) == 0


def test_metrics_full_serialization():
    # Create an observation with all metrics fields
    obs = CmdOutputObservation(
        command='ls',
        content='test.txt',
        metadata=CmdOutputMetadata(exit_code=0),
    )
    metrics = Metrics(model_name='test-model')
    metrics.accumulated_cost = 0.03

    # Add a cost
    cost = Cost(model='test-model', cost=0.02)
    metrics._costs.append(cost)

    # Add a response latency
    latency = ResponseLatency(model='test-model', latency=0.5, response_id='test-id')
    metrics.response_latencies = [latency]

    # Add token usage
    usage = TokenUsage(
        model='test-model',
        prompt_tokens=10,
        completion_tokens=20,
        cache_read_tokens=0,
        cache_write_tokens=0,
        response_id='test-id',
    )
    metrics.token_usages = [usage]

    obs._llm_metrics = metrics

    # Test serialization
    serialized = event_to_dict(obs)
    assert 'llm_metrics' in serialized
    metrics_dict = serialized['llm_metrics']
    assert metrics_dict['accumulated_cost'] == 0.03
    assert len(metrics_dict['costs']) == 1
    assert metrics_dict['costs'][0]['cost'] == 0.02
    assert len(metrics_dict['response_latencies']) == 1
    assert metrics_dict['response_latencies'][0]['latency'] == 0.5
    assert len(metrics_dict['token_usages']) == 1
    assert metrics_dict['token_usages'][0]['prompt_tokens'] == 10
    assert metrics_dict['token_usages'][0]['completion_tokens'] == 20

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.llm_metrics is not None
    assert deserialized.llm_metrics.accumulated_cost == 0.03
    assert len(deserialized.llm_metrics.costs) == 1
    assert deserialized.llm_metrics.costs[0].cost == 0.02
    assert len(deserialized.llm_metrics.response_latencies) == 1
    assert deserialized.llm_metrics.response_latencies[0].latency == 0.5
    assert len(deserialized.llm_metrics.token_usages) == 1
    assert deserialized.llm_metrics.token_usages[0].prompt_tokens == 10
    assert deserialized.llm_metrics.token_usages[0].completion_tokens == 20


def test_metrics_none_serialization():
    # Test when metrics is None
    obs = CmdOutputObservation(
        command='ls',
        content='test.txt',
        metadata=CmdOutputMetadata(exit_code=0),
    )
    obs._llm_metrics = None

    # Test serialization
    serialized = event_to_dict(obs)
    assert 'llm_metrics' not in serialized

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.llm_metrics is None


def test_action_risk_serialization():
    # Test action with security risk
    action = CmdRunAction(command='rm -rf /tmp/test')
    action.security_risk = ActionSecurityRisk.HIGH

    # Test serialization
    serialized = event_to_dict(action)
    assert 'security_risk' in serialized['args']
    assert serialized['args']['security_risk'] == ActionSecurityRisk.HIGH.value

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.security_risk == ActionSecurityRisk.HIGH

    # Test action with no security risk
    action = CmdRunAction(command='ls')
    # Don't set action_risk

    # Test serialization
    serialized = event_to_dict(action)
    assert 'security_risk' in serialized['args']
    assert serialized['args']['security_risk'] == ActionSecurityRisk.UNKNOWN.value

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.security_risk == ActionSecurityRisk.UNKNOWN
