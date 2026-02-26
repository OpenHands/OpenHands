import os

from openhands.core.logger import openhands_logger as logger


class UserVerifier:
    def __init__(self) -> None:
        logger.debug('Initializing UserVerifier')
        self.file_users: list[str] | None = None

        # Initialize from environment variables
        self._init_file_users()

    def _init_file_users(self) -> None:
        """Load users from text file if configured."""
        waitlist = os.getenv('GITHUB_USER_LIST_FILE')
        if not waitlist:
            logger.debug('GITHUB_USER_LIST_FILE not configured')
            return

        if not os.path.exists(waitlist):
            logger.error(f'User list file not found: {waitlist}')
            raise FileNotFoundError(f'User list file not found: {waitlist}')

        try:
            with open(waitlist, 'r') as f:
                self.file_users = [line.strip().lower() for line in f if line.strip()]
            logger.info(
                f'Successfully loaded {len(self.file_users)} users from {waitlist}'
            )
        except Exception:
            logger.exception(f'Error reading user list file {waitlist}')

    def is_active(self) -> bool:
        if os.getenv('DISABLE_WAITLIST', '').lower() == 'true':
            logger.info('Waitlist disabled via DISABLE_WAITLIST env var')
            return False
        return bool(self.file_users)

    def is_user_allowed(self, username: str) -> bool:
        """Check if user is allowed based on file and/or sheet configuration."""
        logger.debug(f'Checking if GitHub user {username} is allowed')
        if self.file_users:
            if username.lower() in self.file_users:
                logger.debug(f'User {username} found in text file allowlist')
                return True
            logger.debug(f'User {username} not found in text file allowlist')

        logger.debug(f'User {username} not found in any allowlist')
        return False


user_verifier = UserVerifier()
