from datetime import datetime, timezone

from openhands.app_server.integrations.bitbucket.service.base import BitBucketMixinBase
from openhands.app_server.integrations.service_types import Comment, RequestMethod


class BitBucketResolverMixin(BitBucketMixinBase):
    """Helper methods used for the Bitbucket Cloud Resolver."""

    async def get_pr_title_and_body(
        self, workspace: str, repo_slug: str, pr_id: int
    ) -> tuple[str, str]:
        """Get the title and description of a pull request.

        Args:
            workspace: Bitbucket workspace slug.
            repo_slug: Repository slug within the workspace.
            pr_id: Pull request id.

        Returns:
            ``(title, description)`` — empty strings when missing on the
            upstream payload.
        """
        url = (
            f'{self.BASE_URL}/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}'
        )
        response, _ = await self._make_request(url)
        title = response.get('title') or ''
        body = response.get('description') or ''
        return title, body

    async def get_pr_comments(
        self,
        workspace: str,
        repo_slug: str,
        pr_id: int,
        max_comments: int = 10,
    ) -> list[Comment]:
        """Get top-level (non-inline, non-deleted) comments for a pull request.

        Walks Bitbucket Cloud's ``next`` pagination links until the page count
        is exhausted or ``max_comments`` non-skipped comments have been
        collected.
        """
        url = (
            f'{self.BASE_URL}/repositories/{workspace}/{repo_slug}'
            f'/pullrequests/{pr_id}/comments'
        )
        all_raw: list[dict] = []
        params: dict | None = {'pagelen': 100}

        while url and len(all_raw) < max_comments:
            response, _ = await self._make_request(url, params)
            for item in response.get('values', []):
                if item.get('deleted'):
                    continue
                if item.get('inline'):
                    continue
                all_raw.append(item)

            url = response.get('next')
            # Subsequent next URLs already carry their own query string.
            params = None

        return self._process_raw_comments(all_raw, max_comments)

    def _process_raw_comments(
        self, comments: list[dict], max_comments: int = 10
    ) -> list[Comment]:
        """Convert raw Bitbucket Cloud comment dicts to :class:`Comment`s.

        Sorted by ``created_on`` ascending; only the most recent
        ``max_comments`` are kept.
        """
        all_comments: list[Comment] = []
        for raw in comments:
            created_at = _parse_bb_datetime(raw.get('created_on'))
            updated_at = _parse_bb_datetime(raw.get('updated_on'))
            user = raw.get('user') or {}
            author = (
                user.get('display_name')
                or user.get('nickname')
                or user.get('account_id')
                or 'unknown'
            )
            body = (raw.get('content') or {}).get('raw', '') or ''
            all_comments.append(
                Comment(
                    id=str(raw.get('id', 'unknown')),
                    body=self._truncate_comment(body),
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
        workspace: str,
        repo_slug: str,
        pr_id: int,
        body: str,
        *,
        parent_comment_id: int | None = None,
        inline: dict | None = None,
    ) -> None:
        """Post a comment back to a pull request.

        When ``parent_comment_id`` is given, the comment becomes a threaded
        reply to that parent. When ``inline`` is given, the comment is
        attached to the supplied file/line.
        """
        url = (
            f'{self.BASE_URL}/repositories/{workspace}/{repo_slug}'
            f'/pullrequests/{pr_id}/comments'
        )
        payload: dict = {'content': {'raw': body}}
        if parent_comment_id is not None:
            payload['parent'] = {'id': parent_comment_id}
        if inline is not None:
            payload['inline'] = inline

        await self._make_request(url, params=payload, method=RequestMethod.POST)

    async def user_has_write_access(self, workspace: str, repo_slug: str) -> bool:
        """Return True when the authenticated user has ``write`` or ``admin``
        permission on ``workspace/repo_slug``.

        Bitbucket Cloud has no per-pull-request permission API, so this is
        the workspace-level analog used by the resolver to gate job creation.
        """
        url = f'{self.BASE_URL}/user/permissions/repositories'
        params = {'q': f'repository.full_name="{workspace}/{repo_slug}"'}
        response, _ = await self._make_request(url, params)
        for entry in response.get('values', []):
            if entry.get('permission') in ('write', 'admin'):
                return True
        return False

    async def user_has_write_access_for(
        self, workspace: str, repo_slug: str, selected_user_id: str
    ) -> bool:
        """Return True when ``selected_user_id`` is a workspace ``owner`` or
        ``collaborator`` on ``workspace`` — Bitbucket Cloud's
        workspace-level analog of "has write access".

        ``repo_slug`` is currently unused. The strict per-repo check at
        ``/repositories/{ws}/{repo}/permissions-config/users/{id}``
        requires ``repository:admin`` OAuth scope, which the OpenHands
        Bitbucket OAuth grant does not request; ``/workspaces/{ws}/permissions``
        is the closest analog that works with the existing ``account``
        scope. The parameter is kept on the signature so that callers
        do not need to change if/when admin scope is added and a
        repo-level check is restored.

        ``selected_user_id`` may be a Bitbucket Cloud ``account_id``
        (``712020:...``) or a UUID with or without braces — both filter
        formats are routed correctly to the upstream ``q`` parameter.
        """
        if ':' in selected_user_id:
            q = f'user.account_id="{selected_user_id}"'
        elif selected_user_id:
            uuid_value = selected_user_id
            if not uuid_value.startswith('{'):
                uuid_value = '{' + uuid_value + '}'
            q = f'user.uuid="{uuid_value}"'
        else:
            return False

        url = f'{self.BASE_URL}/workspaces/{workspace}/permissions'
        try:
            response, _ = await self._make_request(url, {'q': q})
        except Exception:
            return False
        for entry in response.get('values', []):
            if entry.get('permission') in ('owner', 'collaborator'):
                return True
        return False


def _parse_bb_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    # Bitbucket Cloud uses ISO 8601 with timezone offset; tolerate trailing 'Z'.
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
