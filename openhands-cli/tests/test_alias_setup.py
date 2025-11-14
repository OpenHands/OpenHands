"""Tests for alias setup functionality in OpenHands V1 CLI."""

import os
import platform
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from openhands_cli.user_actions.alias_setup import AliasSetup, run_alias_setup


class TestAliasSetup:
    """Test cases for AliasSetup class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alias_setup = AliasSetup()
    
    def test_detect_shell_zsh(self):
        """Test shell detection for zsh."""
        with patch.dict(os.environ, {'SHELL': '/bin/zsh'}):
            alias_setup = AliasSetup()
            assert alias_setup.shell_type == 'zsh'
    
    def test_detect_shell_bash(self):
        """Test shell detection for bash."""
        with patch.dict(os.environ, {'SHELL': '/bin/bash'}):
            alias_setup = AliasSetup()
            assert alias_setup.shell_type == 'bash'
    
    def test_detect_shell_fish(self):
        """Test shell detection for fish."""
        with patch.dict(os.environ, {'SHELL': '/usr/local/bin/fish'}):
            alias_setup = AliasSetup()
            assert alias_setup.shell_type == 'fish'
    
    def test_detect_shell_default(self):
        """Test shell detection with unknown shell defaults to bash."""
        with patch.dict(os.environ, {'SHELL': '/unknown/shell'}, clear=True):
            alias_setup = AliasSetup()
            assert alias_setup.shell_type == 'bash'
    
    @patch('platform.system')
    def test_detect_shell_windows(self, mock_system):
        """Test shell detection on Windows."""
        mock_system.return_value = 'Windows'
        alias_setup = AliasSetup()
        assert alias_setup.shell_type == 'powershell'
    
    @patch('subprocess.run')
    def test_check_uv_installed_success(self, mock_run):
        """Test successful uv installation check."""
        mock_run.return_value.returncode = 0
        assert self.alias_setup._check_uv_installed() is True
        mock_run.assert_called_once_with(
            ['uv', '--version'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
    
    @patch('subprocess.run')
    def test_check_uv_installed_not_found(self, mock_run):
        """Test uv not installed."""
        mock_run.side_effect = FileNotFoundError()
        assert self.alias_setup._check_uv_installed() is False
    
    @patch('subprocess.run')
    def test_check_uv_installed_timeout(self, mock_run):
        """Test uv installation check timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('uv', 5)
        assert self.alias_setup._check_uv_installed() is False
    
    def test_get_shell_profile_path_existing(self):
        """Test getting shell profile path when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            home_path = Path(temp_dir)
            self.alias_setup.home_dir = home_path
            self.alias_setup.shell_type = 'bash'
            
            # Create .bashrc file
            bashrc_path = home_path / '.bashrc'
            bashrc_path.touch()
            
            result = self.alias_setup._get_shell_profile_path()
            assert result == bashrc_path
    
    def test_get_shell_profile_path_nonexisting(self):
        """Test getting shell profile path when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            home_path = Path(temp_dir)
            self.alias_setup.home_dir = home_path
            self.alias_setup.shell_type = 'bash'
            
            result = self.alias_setup._get_shell_profile_path()
            # Should return first profile even if it doesn't exist
            assert result == home_path / '.bashrc'
    
    def test_format_alias_command_bash(self):
        """Test alias command formatting for bash/zsh."""
        self.alias_setup.shell_type = 'bash'
        result = self.alias_setup._format_alias_command('test', 'echo hello')
        assert result == 'alias test="echo hello"'
    
    def test_format_alias_command_fish(self):
        """Test alias command formatting for fish."""
        self.alias_setup.shell_type = 'fish'
        result = self.alias_setup._format_alias_command('test', 'echo hello')
        assert result == 'alias test "echo hello"'
    
    def test_format_alias_command_powershell(self):
        """Test alias command formatting for PowerShell."""
        self.alias_setup.shell_type = 'powershell'
        result = self.alias_setup._format_alias_command('test', 'echo hello')
        assert result == 'function test { echo hello $args }'
    
    def test_add_aliases_to_profile_new_file(self):
        """Test adding aliases to a new profile file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / '.bashrc'
            aliases = {'oh': 'openhands', 'test': 'echo test'}
            
            result = self.alias_setup._add_aliases_to_profile(profile_path, aliases)
            assert result is True
            
            # Check file content
            content = profile_path.read_text()
            assert '# OpenHands CLI aliases' in content
            assert 'alias oh="openhands"' in content
            assert 'alias test="echo test"' in content
    
    def test_add_aliases_to_profile_existing_file(self):
        """Test adding aliases to an existing profile file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / '.bashrc'
            profile_path.write_text('# Existing content\nexport PATH=$PATH:/usr/local/bin\n')
            
            aliases = {'oh': 'openhands'}
            result = self.alias_setup._add_aliases_to_profile(profile_path, aliases)
            assert result is True
            
            content = profile_path.read_text()
            assert '# Existing content' in content
            assert '# OpenHands CLI aliases' in content
            assert 'alias oh="openhands"' in content
    
    def test_add_aliases_to_profile_already_exists(self):
        """Test adding aliases when they already exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / '.bashrc'
            profile_path.write_text('# OpenHands CLI aliases\nalias oh="openhands"\n')
            
            aliases = {'oh': 'openhands'}
            result = self.alias_setup._add_aliases_to_profile(profile_path, aliases)
            assert result is False  # Aliases already exist
    
    def test_add_aliases_to_profile_creates_directory(self):
        """Test that alias setup creates parent directory if needed (for fish config)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / '.config' / 'fish' / 'config.fish'
            aliases = {'oh': 'openhands'}
            
            result = self.alias_setup._add_aliases_to_profile(config_path, aliases)
            assert result is True
            assert config_path.exists()
            assert config_path.parent.exists()
    
    def test_alias_configs_structure(self):
        """Test that alias configurations have proper structure."""
        configs = AliasSetup.ALIAS_CONFIGS
        
        # Check both config types exist
        assert 'uv_tool' in configs
        assert 'uvx' in configs
        
        # Check required keys in each config
        for config_name, config in configs.items():
            assert 'name' in config
            assert 'description' in config
            assert 'aliases' in config
            assert 'prerequisites' in config
            assert 'benefits' in config
            
            # Check aliases structure
            aliases = config['aliases']
            assert 'openhands' in aliases
            assert 'oh' in aliases
    
    def test_shell_profiles_structure(self):
        """Test that shell profile mappings are properly defined."""
        profiles = AliasSetup.SHELL_PROFILES
        
        # Check all expected shell types
        expected_shells = ['bash', 'zsh', 'fish', 'powershell']
        for shell in expected_shells:
            assert shell in profiles
            assert isinstance(profiles[shell], list)
            assert len(profiles[shell]) > 0


class TestRunAliasSetup:
    """Test cases for the run_alias_setup function."""
    
    @patch('openhands_cli.user_actions.alias_setup.AliasSetup')
    def test_run_alias_setup_success(self, mock_alias_setup_class):
        """Test successful alias setup."""
        mock_alias_setup = Mock()
        mock_alias_setup.run_alias_setup_flow.return_value = True
        mock_alias_setup_class.return_value = mock_alias_setup
        
        result = run_alias_setup()
        assert result is True
        mock_alias_setup.run_alias_setup_flow.assert_called_once()
    
    @patch('openhands_cli.user_actions.alias_setup.AliasSetup')
    def test_run_alias_setup_failure(self, mock_alias_setup_class):
        """Test alias setup failure."""
        mock_alias_setup = Mock()
        mock_alias_setup.run_alias_setup_flow.return_value = False
        mock_alias_setup_class.return_value = mock_alias_setup
        
        result = run_alias_setup()
        assert result is False


@pytest.fixture
def temp_home(tmp_path):
    """Create a temporary home directory for testing."""
    return tmp_path


def test_integration_bash_setup(temp_home):
    """Integration test for bash alias setup."""
    with patch.dict(os.environ, {'SHELL': '/bin/bash'}):
        alias_setup = AliasSetup()
        alias_setup.home_dir = temp_home
        
        # Mock profile path
        profile_path = temp_home / '.bashrc'
        
        aliases = {'openhands': 'uvx --python 3.12 --from openhands-ai openhands'}
        result = alias_setup._add_aliases_to_profile(profile_path, aliases)
        
        assert result is True
        content = profile_path.read_text()
        assert 'alias openhands="uvx --python 3.12 --from openhands-ai openhands"' in content


def test_integration_fish_setup(temp_home):
    """Integration test for fish alias setup."""
    with patch.dict(os.environ, {'SHELL': '/usr/local/bin/fish'}):
        alias_setup = AliasSetup()
        alias_setup.home_dir = temp_home
        
        # Fish config path
        config_path = temp_home / '.config' / 'fish' / 'config.fish'
        
        aliases = {'oh': 'openhands'}
        result = alias_setup._add_aliases_to_profile(config_path, aliases)
        
        assert result is True
        assert config_path.exists()
        content = config_path.read_text()
        assert 'alias oh "openhands"' in content
