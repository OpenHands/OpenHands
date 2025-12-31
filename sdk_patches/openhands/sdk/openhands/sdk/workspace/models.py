"""Pydantic models for workspace operation results and build types."""

from typing import Literal

from pydantic import BaseModel, Field


TargetType = Literal["binary", "binary-minimal", "source", "source-minimal"]
PlatformType = Literal["linux/amd64", "linux/arm64"]


class CommandResult(BaseModel):
    """Result of executing a command in the workspace."""

    command: str = Field(description="The command that was executed")
    exit_code: int = Field(description="Exit code of the command")
    stdout: str = Field(description="Standard output from the command")
    stderr: str = Field(description="Standard error from the command")
    timeout_occurred: bool = Field(
        description="Whether the command timed out during execution"
    )


class FileOperationResult(BaseModel):
    """Result of a file upload or download operation."""

    success: bool = Field(description="Whether the operation was successful")
    source_path: str = Field(description="Path to the source file")
    destination_path: str = Field(description="Path to the destination file")
    file_size: int | None = Field(
        default=None, description="Size of the file in bytes (if successful)"
    )
    error: str | None = Field(
        default=None, description="Error message (if operation failed)"
    )
