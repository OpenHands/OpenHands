"""Tests for the context offloader module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from openhands.core.config.offload_config import OffloadConfig
from openhands.memory.offloader import ContextOffloader, OffloadResult


class TestOffloadConfig:
    """Tests for OffloadConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OffloadConfig()
        assert config.enabled is False
        assert config.max_output_chars == 25000
        assert config.offload_dir == '.openhands/context_offload'
        assert config.preview_head_lines == 15
        assert config.preview_tail_lines == 5
        assert config.cleanup_on_session_end is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OffloadConfig(
            enabled=True,
            max_output_chars=10000,
            offload_dir='/custom/path',
            preview_head_lines=20,
            preview_tail_lines=10,
            cleanup_on_session_end=False,
        )
        assert config.enabled is True
        assert config.max_output_chars == 10000
        assert config.offload_dir == '/custom/path'
        assert config.preview_head_lines == 20
        assert config.preview_tail_lines == 10
        assert config.cleanup_on_session_end is False

    def test_min_max_output_chars(self):
        """Test minimum value for max_output_chars."""
        with pytest.raises(ValidationError):
            OffloadConfig(max_output_chars=500)  # Below minimum of 1000


class TestContextOffloader:
    """Tests for ContextOffloader."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def enabled_offloader(self, temp_workspace):
        """Create an enabled offloader instance."""
        config = OffloadConfig(enabled=True, max_output_chars=1000)
        return ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test_session',
        )

    @pytest.fixture
    def disabled_offloader(self, temp_workspace):
        """Create a disabled offloader instance."""
        config = OffloadConfig(enabled=False)
        return ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test_session',
        )

    def test_offloader_disabled(self, disabled_offloader):
        """Test that disabled offloader doesn't offload."""
        content = 'x' * 50000  # Large content
        assert disabled_offloader.should_offload(content) is False

    def test_should_offload_below_threshold(self, enabled_offloader):
        """Test that small content is not offloaded."""
        content = 'small content'
        assert enabled_offloader.should_offload(content) is False

    def test_should_offload_above_threshold(self, enabled_offloader):
        """Test that large content is offloaded."""
        content = 'x' * 2000  # Above 1000 char threshold
        assert enabled_offloader.should_offload(content) is True

    def test_offload_text(self, enabled_offloader, temp_workspace):
        """Test offloading text content."""
        content = '\n'.join([f'Line {i}' for i in range(100)])
        result = enabled_offloader.offload_text(content, source_type='cmd')

        assert isinstance(result, OffloadResult)
        # original_size is in bytes (UTF-8 encoded)
        assert result.original_size == len(content.encode('utf-8'))
        assert result.offload_type == 'cmd'
        assert Path(result.file_path).exists()

        # Verify file content
        saved_content = Path(result.file_path).read_text(encoding='utf-8')
        assert saved_content == content

        # Verify preview message contains key info
        assert 'offloaded' in result.preview_message.lower()
        assert result.file_path in result.preview_message

    def test_offload_text_unicode_bytes(self, enabled_offloader, temp_workspace):
        """Test that original_size is bytes, not characters (important for Unicode)."""
        # Unicode content: Chinese characters take 3 bytes each in UTF-8
        content = '中文测试内容 ' * 200  # 7 chars per repeat, 200 repeats = 1400 chars
        result = enabled_offloader.offload_text(content, source_type='cmd')

        # original_size should be bytes, not characters
        char_count = len(content)  # 1400 chars
        byte_count = len(content.encode('utf-8'))  # Much larger due to UTF-8 encoding

        assert result.original_size == byte_count
        assert result.original_size > char_count  # Bytes > chars for Chinese text

    def test_offload_json(self, enabled_offloader, temp_workspace):
        """Test offloading JSON content."""
        data = {
            'key1': 'value1',
            'key2': ['a', 'b', 'c'],
            'key3': {'nested': 'object'},
        }
        result = enabled_offloader.offload_json(data, source_type='browser_dom')

        assert isinstance(result, OffloadResult)
        assert result.offload_type == 'browser_dom'
        assert Path(result.file_path).exists()

        # Verify file content
        saved_data = json.loads(Path(result.file_path).read_text(encoding='utf-8'))
        assert saved_data == data

    def test_text_preview_short_content(self, temp_workspace):
        """Test preview for content shorter than head+tail lines."""
        config = OffloadConfig(
            enabled=True,
            max_output_chars=1000,
            preview_head_lines=10,
            preview_tail_lines=5,
        )
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )

        # Content with only 5 lines but > 1000 chars (less than head+tail=15)
        content = '\n'.join([f'Line {i} ' + 'x' * 200 for i in range(5)])
        result = offloader.offload_text(content, source_type='cmd')

        # Should use short template (no head/tail split)
        assert 'omitted' not in result.preview_message.lower()

    def test_text_preview_long_content(self, temp_workspace):
        """Test preview for content with many lines."""
        config = OffloadConfig(
            enabled=True,
            max_output_chars=1000,
            preview_head_lines=5,
            preview_tail_lines=3,
        )
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )

        # Content with 100 lines and > 1000 chars
        content = '\n'.join([f'Line {i} ' + 'x' * 20 for i in range(100)])
        result = offloader.offload_text(content, source_type='cmd')

        # Should show head and tail with omitted count
        assert 'omitted' in result.preview_message.lower()
        assert 'Line 0' in result.preview_message  # First line
        assert 'Line 99' in result.preview_message  # Last line

    def test_cleanup(self, enabled_offloader, temp_workspace):
        """Test cleanup of offloaded files."""
        # Create some offloaded files
        content = 'x' * 2000
        result1 = enabled_offloader.offload_text(content, source_type='cmd')
        result2 = enabled_offloader.offload_text(content, source_type='cmd')

        assert Path(result1.file_path).exists()
        assert Path(result2.file_path).exists()

        # Cleanup
        removed = enabled_offloader.cleanup()
        assert removed == 2
        assert not Path(result1.file_path).exists()
        assert not Path(result2.file_path).exists()

    def test_cleanup_disabled(self, temp_workspace):
        """Test that cleanup is skipped when disabled."""
        config = OffloadConfig(
            enabled=True,
            max_output_chars=1000,
            cleanup_on_session_end=False,
        )
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )

        content = 'x' * 2000  # > 1000 chars to trigger offload
        result = offloader.offload_text(content, source_type='cmd')
        assert Path(result.file_path).exists()

        # Cleanup should not remove files
        removed = offloader.cleanup()
        assert removed == 0
        assert Path(result.file_path).exists()

    def test_get_stats(self, enabled_offloader, temp_workspace):
        """Test getting offloader statistics."""
        # Initial stats
        stats = enabled_offloader.get_stats()
        assert stats['enabled'] is True
        assert stats['files_count'] == 0
        assert stats['total_size_bytes'] == 0

        # After offloading
        content = 'x' * 2000
        enabled_offloader.offload_text(content, source_type='cmd')

        stats = enabled_offloader.get_stats()
        assert stats['files_count'] == 1
        assert stats['total_size_bytes'] > 0

    def test_size_limit(self, temp_workspace):
        """Test total size limit enforcement."""
        config = OffloadConfig(
            enabled=True,
            max_output_chars=1000,
            max_total_size_mb=1,  # 1MB limit
        )
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )

        # Create content that would exceed limit (2MB)
        large_content = 'x' * (1024 * 1024 * 2)  # 2MB

        with pytest.raises(IOError) as exc_info:
            offloader.offload_text(large_content, source_type='cmd')

        assert 'size limit' in str(exc_info.value).lower()

    def test_unique_filenames(self, enabled_offloader):
        """Test that generated filenames are unique."""
        content = 'x' * 2000
        filenames = set()

        for _ in range(10):
            result = enabled_offloader.offload_text(content, source_type='cmd')
            assert result.file_path not in filenames
            filenames.add(result.file_path)


class TestCmdOutputObservationOffload:
    """Tests for CmdOutputObservation with offloading."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def offloader(self, temp_workspace):
        """Create an offloader instance."""
        config = OffloadConfig(enabled=True, max_output_chars=1000)
        return ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test_session',
        )

    def test_cmd_output_with_offloader_small_content(self, offloader):
        """Test CmdOutputObservation with small content (no offload)."""
        from openhands.events.observation.commands import (
            CmdOutputMetadata,
            CmdOutputObservation,
        )

        obs = CmdOutputObservation(
            content='small output',
            command='echo hello',
            metadata=CmdOutputMetadata(),
            offloader=offloader,
        )

        # Small content should not be offloaded
        assert obs.offloaded_path is None
        assert obs.content == 'small output'

    def test_cmd_output_with_offloader_large_content(self, offloader):
        """Test CmdOutputObservation with large content (should offload)."""
        from openhands.events.observation.commands import (
            CmdOutputMetadata,
            CmdOutputObservation,
        )

        large_content = '\n'.join([f'Line {i}' for i in range(200)])

        obs = CmdOutputObservation(
            content=large_content,
            command='cat large_file',
            metadata=CmdOutputMetadata(),
            offloader=offloader,
        )

        # Large content should be offloaded
        assert obs.offloaded_path is not None
        assert Path(obs.offloaded_path).exists()
        assert obs.original_size == len(large_content)

        # Content should be preview message
        assert 'offloaded' in obs.content.lower()

    def test_cmd_output_without_offloader(self):
        """Test CmdOutputObservation without offloader (uses truncation)."""
        from openhands.events.observation.commands import (
            CmdOutputMetadata,
            CmdOutputObservation,
        )

        # Very large content should be truncated (not offloaded)
        large_content = 'x' * 50000

        obs = CmdOutputObservation(
            content=large_content,
            command='cat huge_file',
            metadata=CmdOutputMetadata(),
            offloader=None,
        )

        # Without offloader, should fall back to truncation
        assert obs.offloaded_path is None
        # Content may be truncated
        assert len(obs.content) <= len(large_content)


class TestSecurityFeatures:
    """Tests for security features (path traversal prevention)."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_session_id_sanitization_valid(self, temp_workspace):
        """Test that valid session_id passes through."""
        config = OffloadConfig(enabled=True)
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='valid-session_123',
        )
        assert offloader.session_id == 'valid-session_123'

    def test_session_id_sanitization_with_slashes(self, temp_workspace):
        """Test that session_id with path separators gets sanitized."""
        config = OffloadConfig(enabled=True)
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='../../../etc/passwd',
        )
        # Should be sanitized (slashes removed, then hashed if invalid)
        assert '/' not in offloader.session_id
        assert '\\' not in offloader.session_id
        assert '..' not in offloader.session_id

    def test_session_id_sanitization_empty_raises(self, temp_workspace):
        """Test that empty session_id raises ValueError."""
        config = OffloadConfig(enabled=True)
        with pytest.raises(ValueError) as exc_info:
            ContextOffloader(
                config=config,
                workspace_dir=temp_workspace,
                session_id='',
            )
        assert 'cannot be empty' in str(exc_info.value)

    def test_offload_dir_sanitization(self, temp_workspace):
        """Test that offload_dir with path traversal gets sanitized."""
        config = OffloadConfig(
            enabled=True,
            offload_dir='../../../tmp/malicious',
        )
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )
        # Offload dir should be within workspace
        workspace_path = Path(temp_workspace).resolve()
        assert str(offloader.offload_dir.resolve()).startswith(str(workspace_path))

    def test_source_type_validation(self, temp_workspace):
        """Test that invalid source_type falls back to 'file'."""
        config = OffloadConfig(enabled=True, max_output_chars=1000)
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )
        # Should not raise, should fall back to 'file'
        content = 'x' * 2000
        result = offloader.offload_text(content, source_type='../../malicious')
        assert 'file_' in result.file_path
        assert result.offload_type == 'file'

    def test_source_type_valid(self, temp_workspace):
        """Test that valid source_type is preserved."""
        config = OffloadConfig(enabled=True, max_output_chars=1000)
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )
        content = 'x' * 2000
        result = offloader.offload_text(content, source_type='cmd')
        assert 'cmd_' in result.file_path
        assert result.offload_type == 'cmd'


class TestRetentionCleanup:
    """Tests for retention-based cleanup."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_cleanup_expired_files(self, temp_workspace):
        """Test that expired files are cleaned up based on retention_hours."""
        import os
        import time

        config = OffloadConfig(
            enabled=True,
            max_output_chars=1000,
            cleanup_on_session_end=False,
            retention_hours=1,  # 1 hour retention
        )

        # Create offloader and add some files
        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )

        content = 'x' * 2000
        result = offloader.offload_text(content, source_type='cmd')
        file_path = Path(result.file_path)
        assert file_path.exists()

        # Set file modification time to 2 hours ago
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(file_path, (old_time, old_time))

        # Run cleanup - should remove the old file
        removed = offloader.cleanup_expired_files()
        assert removed == 1
        assert not file_path.exists()

    def test_cleanup_expired_files_keeps_recent(self, temp_workspace):
        """Test that recent files are kept during retention cleanup."""
        config = OffloadConfig(
            enabled=True,
            max_output_chars=1000,
            cleanup_on_session_end=False,
            retention_hours=24,  # 24 hour retention
        )

        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )

        content = 'x' * 2000
        result = offloader.offload_text(content, source_type='cmd')
        file_path = Path(result.file_path)
        assert file_path.exists()

        # Run cleanup - file should be kept (it's recent)
        removed = offloader.cleanup_expired_files()
        assert removed == 0
        assert file_path.exists()

    def test_retention_cleanup_disabled_when_session_cleanup_enabled(
        self, temp_workspace
    ):
        """Test that retention cleanup is disabled when cleanup_on_session_end is True."""
        config = OffloadConfig(
            enabled=True,
            max_output_chars=1000,
            cleanup_on_session_end=True,  # Session cleanup enabled
            retention_hours=1,
        )

        offloader = ContextOffloader(
            config=config,
            workspace_dir=temp_workspace,
            session_id='test',
        )

        content = 'x' * 2000
        result = offloader.offload_text(content, source_type='cmd')
        file_path = Path(result.file_path)

        # cleanup_expired_files should return 0 (disabled)
        removed = offloader.cleanup_expired_files()
        assert removed == 0
        assert file_path.exists()
