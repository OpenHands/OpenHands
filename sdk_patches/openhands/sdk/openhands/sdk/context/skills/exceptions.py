class SkillError(Exception):
    """Base exception for all skill errors."""

    pass


class SkillValidationError(SkillError):
    """Raised when there's a validation error in skill metadata."""

    def __init__(self, message: str = "Skill validation failed") -> None:
        super().__init__(message)
