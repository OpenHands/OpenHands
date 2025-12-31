from server.auth.constants import BLOCKED_EMAIL_DOMAINS

from openhands.core.logger import openhands_logger as logger


class DomainBlocker:
    def __init__(self) -> None:
        logger.debug('Initializing DomainBlocker')
        self.blocked_domains: list[str] = BLOCKED_EMAIL_DOMAINS
        if self.blocked_domains:
            logger.info(
                f'Successfully loaded {len(self.blocked_domains)} blocked email domains: {self.blocked_domains}'
            )

    def is_active(self) -> bool:
        """Check if domain blocking is enabled"""
        return bool(self.blocked_domains)

    def _extract_domain(self, email: str) -> str | None:
        """Extract and normalize email domain from email address"""
        if not email:
            return None
        try:
            # Extract domain part after @
            if '@' not in email:
                return None
            domain = email.split('@')[1].strip().lower()
            return domain if domain else None
        except Exception:
            logger.debug(f'Error extracting domain from email: {email}', exc_info=True)
            return None

    def is_domain_blocked(self, email: str) -> bool:
        """Check if email domain is blocked"""
        if not self.is_active():
            return False

        if not email:
            logger.debug('No email provided for domain check')
            return False

        domain = self._extract_domain(email)
        if not domain:
            logger.debug(f'Could not extract domain from email: {email}')
            return False

        is_blocked = domain in self.blocked_domains
        if is_blocked:
            logger.warning(f'Email domain {domain} is blocked for email: {email}')
        else:
            logger.debug(f'Email domain {domain} is not blocked')

        return is_blocked


domain_blocker = DomainBlocker()
