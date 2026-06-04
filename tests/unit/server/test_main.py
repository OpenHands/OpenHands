from unittest.mock import Mock

from openhands.server.__main__ import main


def test_main_uses_default_port(monkeypatch):
    run = Mock()
    monkeypatch.delenv('PORT', raising=False)
    monkeypatch.delenv('port', raising=False)
    monkeypatch.setattr('openhands.server.__main__.uvicorn.run', run)

    main()

    assert run.call_args.kwargs['port'] == 3000


def test_main_uses_port_env(monkeypatch):
    run = Mock()
    monkeypatch.setenv('PORT', '4040')
    monkeypatch.setattr('openhands.server.__main__.uvicorn.run', run)

    main()

    assert run.call_args.kwargs['port'] == 4040


def test_main_keeps_legacy_lowercase_port_env(monkeypatch):
    run = Mock()
    monkeypatch.delenv('PORT', raising=False)
    monkeypatch.setenv('port', '5050')
    monkeypatch.setattr('openhands.server.__main__.uvicorn.run', run)

    main()

    assert run.call_args.kwargs['port'] == 5050
