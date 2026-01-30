from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker


@dataclass
class BlockedEmailDomainStore:
    session_maker: sessionmaker

    def is_domain_blocked(self, domain: str) -> bool:
        """Check if a domain is blocked by querying the database directly.

        This method uses SQL to efficiently check if the domain matches any blocked pattern:
        - TLD patterns (e.g., '.us'): checks if domain ends with the pattern
        - Full domain patterns (e.g., 'example.com'): checks for exact match or subdomain match

        Args:
            domain: The extracted domain from the email (e.g., 'example.com' or 'subdomain.example.com')

        Returns:
            True if the domain is blocked, False otherwise
        """
        with self.session_maker() as session:
            # SQL query that handles both TLD patterns and full domain patterns
            # TLD patterns (starting with '.'): check if domain ends with the pattern
            # Full domain patterns: check for exact match or subdomain match
            # All comparisons are case-insensitive using LOWER() to ensure consistent matching
            query = text("""
                SELECT EXISTS(
                    SELECT 1
                    FROM blocked_email_domains
                    WHERE
                        -- TLD pattern (e.g., '.us') - check if domain ends with it (case-insensitive)
                        (LOWER(domain) LIKE '.%' AND LOWER(:domain) LIKE '%' || LOWER(domain)) OR
                        -- Full domain pattern (e.g., 'example.com')
                        -- Block exact match or subdomains (case-insensitive)
                        (LOWER(domain) NOT LIKE '.%' AND (
                            LOWER(:domain) = LOWER(domain) OR
                            LOWER(:domain) LIKE '%.' || LOWER(domain)
                        ))
                )
            """)
            result = session.execute(query, {'domain': domain}).scalar()
            return bool(result)
