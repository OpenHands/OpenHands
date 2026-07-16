from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / 'scripts' / 'openhands_operator.py'
SPEC = importlib.util.spec_from_file_location('openhands_operator', MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f'Unable to load operator module from {MODULE_PATH}')

operator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = operator
SPEC.loader.exec_module(operator)


class VersionParsingTests(unittest.TestCase):
    def test_parse_version_accepts_common_tool_output(self) -> None:
        self.assertEqual(operator.parse_version('v22.12.0'), (22, 12, 0))
        self.assertEqual(
            operator.parse_version('Poetry (version 2.3.4)'), (2, 3, 4)
        )

    def test_parse_version_returns_none_without_semantic_version(self) -> None:
        self.assertIsNone(operator.parse_version('version unknown'))


class ProviderValidationTests(unittest.TestCase):
    def test_partial_generic_configuration_is_an_error_without_secret_leak(self) -> None:
        secret = 'generic-secret-value'
        results = operator.validate_provider(
            {'LLM_API_KEY': secret}, mode='generic', require_provider=False
        )

        self.assertTrue(any(result.status == 'error' for result in results))
        rendered = operator.render_report(
            operator.ReadinessReport(results=results), as_json=False
        )
        self.assertIn('LLM_MODEL', rendered)
        self.assertNotIn(secret, rendered)

    def test_missing_provider_is_warning_unless_required(self) -> None:
        optional = operator.validate_provider(
            {}, mode='auto', require_provider=False
        )
        required = operator.validate_provider(
            {}, mode='auto', require_provider=True
        )

        self.assertEqual(optional[0].status, 'warning')
        self.assertEqual(required[0].status, 'error')

    def test_complete_opencode_go_configuration_maps_to_child_environment(self) -> None:
        source = {
            'OPENCODE_GO_MODEL': 'example-model',
            'OPENCODE_GO_BASE_URL': 'https://provider.example/v1',
            'OPENCODE_GO_API_KEY': 'opencode-secret-value',
            'UNCHANGED': 'yes',
        }

        results = operator.validate_provider(
            source, mode='opencode-go', require_provider=True
        )
        child = operator.build_child_environment(source, provider_mode='opencode-go')

        self.assertFalse(any(result.status == 'error' for result in results))
        self.assertEqual(child['LLM_MODEL'], 'openai/example-model')
        self.assertEqual(child['LLM_BASE_URL'], source['OPENCODE_GO_BASE_URL'])
        self.assertEqual(child['LLM_API_KEY'], source['OPENCODE_GO_API_KEY'])
        self.assertEqual(child['UNCHANGED'], 'yes')

    def test_existing_model_provider_prefix_is_preserved(self) -> None:
        source = {
            'OPENCODE_GO_MODEL': 'openai/example-model',
            'OPENCODE_GO_BASE_URL': 'https://provider.example/v1',
            'OPENCODE_GO_API_KEY': 'secret',
        }

        child = operator.build_child_environment(source, provider_mode='opencode-go')

        self.assertEqual(child['LLM_MODEL'], 'openai/example-model')

    def test_json_report_contains_counts_and_no_secret(self) -> None:
        secret = 'json-secret-value'
        results = operator.validate_provider(
            {
                'OPENCODE_GO_MODEL': 'example-model',
                'OPENCODE_GO_API_KEY': secret,
            },
            mode='opencode-go',
            require_provider=True,
        )

        rendered = operator.render_report(
            operator.ReadinessReport(results=results), as_json=True
        )
        payload = json.loads(rendered)

        self.assertGreaterEqual(payload['summary']['errors'], 1)
        self.assertNotIn(secret, rendered)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_creates_workspace_and_local_config_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            workspace = repo_root / 'workspace'
            template = repo_root / 'config.template.toml'
            config = repo_root / 'config.toml'
            template.write_text('[core]\n', encoding='utf-8')

            first_actions = operator.bootstrap_workspace(
                repo_root=repo_root,
                workspace=workspace,
                create_config=True,
            )
            config.write_text('custom = true\n', encoding='utf-8')
            second_actions = operator.bootstrap_workspace(
                repo_root=repo_root,
                workspace=workspace,
                create_config=True,
            )

            self.assertTrue(workspace.is_dir())
            self.assertEqual(config.read_text(encoding='utf-8'), 'custom = true\n')
            self.assertTrue(any('Created workspace' in item for item in first_actions))
            self.assertTrue(any('already exists' in item for item in second_actions))


class CliTests(unittest.TestCase):
    def test_doctor_returns_error_for_partial_provider_without_printing_secret(self) -> None:
        secret = 'doctor-secret-value'
        output = StringIO()
        with (
            patch.dict(os.environ, {'LLM_API_KEY': secret}, clear=True),
            redirect_stdout(output),
        ):
            exit_code = operator.main(
                [
                    'doctor',
                    '--runtime',
                    'local',
                    '--provider',
                    'generic',
                    '--skip-system-checks',
                    '--skip-port-checks',
                    '--json',
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertNotIn(secret, output.getvalue())
        self.assertIn('LLM_MODEL', output.getvalue())

    def test_start_dry_run_is_sanitized_and_does_not_execute_make(self) -> None:
        secret = 'start-secret-value'
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / 'workspace'
            workspace.mkdir()
            output = StringIO()
            environment = {
                'OPENCODE_GO_MODEL': 'example-model',
                'OPENCODE_GO_BASE_URL': 'https://provider.example/v1',
                'OPENCODE_GO_API_KEY': secret,
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(operator, 'execute_command') as execute_command,
                redirect_stdout(output),
            ):
                exit_code = operator.main(
                    [
                        'start',
                        '--runtime',
                        'local',
                        '--provider',
                        'opencode-go',
                        '--workspace',
                        str(workspace),
                        '--skip-system-checks',
                        '--skip-port-checks',
                        '--dry-run',
                    ]
                )

            self.assertEqual(exit_code, 0)
            execute_command.assert_not_called()
            self.assertIn('LLM_API_KEY=<set>', output.getvalue())
            self.assertIn('INSTALL_DOCKER=0', output.getvalue())
            self.assertNotIn(secret, output.getvalue())


class RemoteAccessSafetyTests(unittest.TestCase):
    def test_remote_binding_requires_explicit_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            blocked = operator.collect_readiness(
                env={},
                runtime='local',
                workspace=workspace,
                backend_host='0.0.0.0',
                backend_port=3000,
                frontend_host='0.0.0.0',
                frontend_port=3001,
                provider_mode='none',
                require_provider=False,
                strict=False,
                skip_system_checks=True,
                skip_port_checks=True,
                allow_remote_access=False,
            )
            allowed = operator.collect_readiness(
                env={},
                runtime='local',
                workspace=workspace,
                backend_host='0.0.0.0',
                backend_port=3000,
                frontend_host='0.0.0.0',
                frontend_port=3001,
                provider_mode='none',
                require_provider=False,
                strict=False,
                skip_system_checks=True,
                skip_port_checks=True,
                allow_remote_access=True,
            )

        self.assertTrue(
            any(
                result.name == 'Remote access' and result.status == 'error'
                for result in blocked.results
            )
        )
        self.assertFalse(
            any(
                result.name == 'Remote access' and result.status == 'error'
                for result in allowed.results
            )
        )


if __name__ == '__main__':
    unittest.main()
