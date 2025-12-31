from openhands.core.schema.exit_reason import ExitReason


def test_exit_reason_enum_values():
    assert ExitReason.INTENTIONAL.value == 'intentional'
    assert ExitReason.INTERRUPTED.value == 'interrupted'
    assert ExitReason.ERROR.value == 'error'


def test_exit_reason_enum_names():
    assert ExitReason['INTENTIONAL'] == ExitReason.INTENTIONAL
    assert ExitReason['INTERRUPTED'] == ExitReason.INTERRUPTED
    assert ExitReason['ERROR'] == ExitReason.ERROR


def test_exit_reason_str_representation():
    assert str(ExitReason.INTENTIONAL) == 'ExitReason.INTENTIONAL'
    assert repr(ExitReason.ERROR) == "<ExitReason.ERROR: 'error'>"
