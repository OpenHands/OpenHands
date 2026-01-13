from storage.blocked_email_domain_store import BlockedEmailDomainStore
from storage.database import session_maker

from openhands.core.logger import openhands_logger as logger


class DomainBlocker:
    def __init__(self, store: BlockedEmailDomainStore) -> None:
        logger.debug('Initializing DomainBlocker')
        self.store = store

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
        """Check if email domain is blocked by querying the database directly via SQL.

        Supports blocking:
        - Exact domains: 'example.com' blocks 'user@example.com'
        - Subdomains: 'example.com' blocks 'user@subdomain.example.com'
        - TLDs: '.us' blocks 'user@company.us' and 'user@subdomain.company.us'

        The blocking logic is handled efficiently in SQL, avoiding the need to load
        all blocked domains into memory.
        """
        if not email:
            logger.debug('No email provided for domain check')
            return False

        domain = self._extract_domain(email)
        if not domain:
            logger.debug(f'Could not extract domain from email: {email}')
            return False

        try:
            # Query database directly via SQL to check if domain is blocked
            is_blocked = self.store.is_domain_blocked(domain)

            if is_blocked:
                logger.warning(f'Email domain {domain} is blocked for email: {email}')
            else:
                logger.debug(f'Email domain {domain} is not blocked')

            return is_blocked
        except Exception as e:
            logger.error(
                f'Error checking if domain is blocked for email {email}: {e}',
                exc_info=True,
            )
            # Fail-safe: if database query fails, don't block (allow auth to proceed)
            return False


# Initialize store and domain blocker
_store = BlockedEmailDomainStore(session_maker=session_maker)
domain_blocker = DomainBlocker(store=_store)
