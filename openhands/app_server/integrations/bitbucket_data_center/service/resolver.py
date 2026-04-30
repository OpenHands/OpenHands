from datetime import datetime, timezone

from openhands.app_server.integrations.bitbucket_data_center.service.base import (
    BitbucketDCMixinBase,
)
from openhands.app_server.integrations.service_types import Comment, RequestMethod


class BitbucketDCResolverMixin(BitbucketDCMixinBase):
    """
    Helper methods used for the Bitbucket Data Center Resolver
    """

    async def get_pr_title_and_body(
        self, owner: str, repo_slug: str, pr_id: int
    ) -> tuple[str, str]:
        """Get the title and body of a pull request.

        Args:
            owner: Project key (e.g. 'PROJ')
            repo_slug: Repository slug
            pr_id: Pull request ID

        Returns:
            A tuple of (title, body)
        """
        url = (
            f'{self.BASE_URL}/projects/{owner}/repos/{repo_slug}/pull-requests/{pr_id}'
        )
        response, _ = await self._make_request(url)
        title = response.get('title') or ''
        body = response.get('description') or ''
        return title, body

    async def get_pr_comments(
        self, owner: str, repo_slug: str, pr_id: int, max_comments: int = 10
    ) -> list[Comment]:
        """Get comments for a pull request.

        Uses the pull-requests/{id}/activities endpoint, filtering for
        COMMENTED actions — the same approach used by the resolver interface.

        Args:
            owner: Project key (e.g. 'PROJ')
            repo_slug: Repository slug
            pr_id: Pull request ID
            max_comments: Maximum number of comments to retrieve

        Returns:
            List of Comment objects ordered by creation date
        """
        url = f'{self.BASE_URL}/projects/{owner}/repos/{repo_slug}/pull-requests/{pr_id}/activities'
        all_raw: list[dict] = []

        params: dict = {'limit': 100, 'start': 0}
        while len(all_raw) < max_comments:
            response, _ = await self._make_request(url, params)
            for activity in response.get('values', []):
                if activity.get('action') == 'COMMENTED':
                    comment = activity.get('comment', {})
                    if comment:
                        all_raw.append(comment)

            if response.get('isLastPage', True):
                break

            next_start = response.get('nextPageStart')
            if next_start is None:
                break
            params = {'limit': 100, 'start': next_start}

        return self._process_raw_comments(all_raw, max_comments)

    def _process_raw_comments(
        self, comments: list, max_comments: int = 10
    ) -> list[Comment]:
        """Convert raw Bitbucket DC comment dicts to Comment objects."""
        all_comments: list[Comment] = []
        for comment_data in comments:
            # Bitbucket DC activities use epoch milliseconds for createdDate/updatedDate
            created_ms = comment_data.get('createdDate')
            updated_ms = comment_data.get('updatedDate')

            created_at = (
                datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
                if created_ms is not None
                else datetime.fromtimestamp(0, tz=timezone.utc)
            )
            updated_at = (
                datetime.fromtimestamp(updated_ms / 1000, tz=timezone.utc)
                if updated_ms is not None
                else datetime.fromtimestamp(0, tz=timezone.utc)
            )

            author = (
                comment_data.get('author', {}).get('slug')
                or comment_data.get('author', {}).get('name')
                or 'unknown'
            )

            all_comments.append(
                Comment(
                    id=str(comment_data.get('id', 'unknown')),
                    body=self._truncate_comment(comment_data.get('text', '')),
                    author=author,
                    created_at=created_at,
                    updated_at=updated_at,
                    system=False,
                )
            )

        all_comments.sort(key=lambda c: c.created_at)
        return all_comments[-max_comments:]

    async def reply_to_pr_comment(
        self,
        owner: str,
        repo_slug: str,
        pr_id: int,
        body: str,
        *,
        parent_comment_id: int | None = None,
        anchor: dict | None = None,
    ) -> None:
        """Post a comment back to a Bitbucket Data Center pull request.

        ``parent_comment_id`` makes the comment a threaded reply. ``anchor``
        attaches the comment to a specific file/line — Bitbucket DC's
        equivalent of Cloud's ``inline`` block. Anchor shape:
        ``{path, line, lineType: 'ADDED'|'CONTEXT'|'REMOVED', fileType: 'TO'|'FROM'}``.
        """
        url = (
            f'{self.BASE_URL}/projects/{owner}/repos/{repo_slug}'
            f'/pull-requests/{pr_id}/comments'
        )
        payload: dict = {'text': body}
        if parent_comment_id is not None:
            payload['parent'] = {'id': parent_comment_id}
        if anchor is not None:
            payload['anchor'] = anchor

        await self._make_request(url, params=payload, method=RequestMethod.POST)

    async def user_has_write_access(self, owner: str, repo_slug: str) -> bool:
        """Self-permission analog of :meth:`user_has_write_access_for`.

        Same policy: any identifiable user is allowed. See that method's
        docstring for the rationale.
        """
        user = await self.get_user()
        slug = user.login or self.user_id
        return await self.user_has_write_access_for(owner, repo_slug, slug or '')

    async def user_has_write_access_for(
        self, owner: str, repo_slug: str, selected_user: str
    ) -> bool:
        """Return True when ``selected_user`` is allowed to trigger the
        resolver on ``owner/repo_slug``.

        Bitbucket Data Center's OAuth 2.0 provider does not expose any
        permission-introspection endpoint that works without ``REPO_ADMIN``
        scope — every variant of ``/permissions/users`` (repo, project,
        global) requires admin per Atlassian's docs. The OpenHands OAuth
        grant requests ``REPO_WRITE``, which is sufficient for reading
        PRs and posting comments but not for the strict introspection
        check. Requiring admin scope on every install would force each
        customer's BBDC admin to grant the integration a permission it
        never actually needs to do its work.

        Mirroring the Cloud counterpart's "downgrade to a non-admin
        endpoint" pattern (see
        ``openhands/app_server/integrations/bitbucket/service/resolver.py``),
        we trust the act of webhook installation as the implicit
        endorsement of anyone who can interact with PRs in the repo. The
        trigger is gated by:

        1. **Webhook signature verification** — only payloads HMAC-signed
           with the per-repo shared secret reach this code.
        2. **Identifiable actor** — ``selected_user`` must be a non-empty
           slug or numeric id extracted from the payload.

        Anyone with at least ``REPO_READ`` can comment on a PR; that's
        the floor we accept. Customers needing finer-grained gating
        should layer a separate allowlist on top — the resolver doesn't
        ship one today.
        """
        return bool(selected_user)
