import json

import pytest

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.browse import BrowseInteractiveAction
from openhands.events.observation.browse import BrowserOutputObservation
from tests.runtime.conftest import _close_test_runtime, _load_runtime


def has_miniwob():
    try:
        import importlib.util

        # try to find this browser environment, if it was installed
        spec = importlib.util.find_spec('browsergym.miniwob')
        if spec is None:
            return False

        # try to import this environment
        importlib.util.module_from_spec(spec)
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not has_miniwob(),
    reason='Requires browsergym-miniwob package to be installed',
)
def test_browsergym_eval_env(runtime_cls, temp_dir):
    runtime, config = _load_runtime(
        temp_dir,
        runtime_cls=runtime_cls,
        run_as_openhands=False,  # need root permission to access file
        base_container_image='xingyaoww/od-eval-miniwob:v1.0',
        browsergym_eval_env='browsergym/miniwob.choose-list',
        force_rebuild_runtime=True,
        enable_browser=True,
    )
    from openhands.runtime.browser.browser_env import (
        BROWSER_EVAL_GET_GOAL_ACTION,
        BROWSER_EVAL_GET_REWARDS_ACTION,
    )

    # Test browse
    action = BrowseInteractiveAction(
        browser_actions=BROWSER_EVAL_GET_GOAL_ACTION, return_axtree=False
    )
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, BrowserOutputObservation)
    assert not obs.error
    assert 'Select' in obs.content
    assert 'from the list and click Submit' in obs.content

    # Make sure the browser can produce observation in eval env
    action = BrowseInteractiveAction(browser_actions='noop()', return_axtree=False)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert (
        obs.url.strip()
        == 'file:///miniwob-plusplus/miniwob/html/miniwob/choose-list.html'
    )

    # Make sure the rewards are working
    action = BrowseInteractiveAction(
        browser_actions=BROWSER_EVAL_GET_REWARDS_ACTION, return_axtree=False
    )
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert json.loads(obs.content) == [0.0]

    _close_test_runtime(runtime)
