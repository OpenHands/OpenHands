import asyncio

from integrations.github.data_collector import GitHubDataCollector
from storage.openhands_pr import OpenhandsPR
from storage.openhands_pr_store import OpenhandsPRStore

from openhands.core.logger import openhands_logger as logger

PROCESS_AMOUNT = 50
MAX_RETRIES = 3

store = OpenhandsPRStore.get_instance()
data_collector = GitHubDataCollector()


async def get_unprocessed_prs() -> list[OpenhandsPR]:
    """
    Get unprocessed PR entries from the OpenhandsPR table.

    Args:
        limit: Maximum number of PRs to retrieve (default: 50)

    Returns:
        List of OpenhandsPR objects that need processing
    """
    unprocessed_prs = await store.get_unprocessed_prs(PROCESS_AMOUNT, MAX_RETRIES)
    logger.info(f'Retrieved {len(unprocessed_prs)} unprocessed PRs for enrichment')
    return unprocessed_prs


async def process_pr(pr: OpenhandsPR):
    """
    Process a single PR to enrich its data.
    """

    logger.info(f'Processing PR #{pr.pr_number} from repo {pr.repo_name}')
    await data_collector.save_full_pr(pr)
    await store.increment_process_attempts(pr.repo_id, pr.pr_number)


async def main():
    """
    Main function to retrieve and process unprocessed PRs.
    """
    logger.info('Starting PR data enrichment process')

    # Get unprocessed PRs
    unprocessed_prs = await get_unprocessed_prs()
    logger.info(f'Found {len(unprocessed_prs)} PRs to process')

    # Process each PR
    for pr in unprocessed_prs:
        try:
            await process_pr(pr)
            logger.info(
                f'Successfully processed PR #{pr.pr_number} from repo {pr.repo_name}'
            )
        except Exception as e:
            logger.exception(
                f'Error processing PR #{pr.pr_number} from repo {pr.repo_name}: {str(e)}'
            )

    logger.info('PR data enrichment process completed')


if __name__ == '__main__':
    asyncio.run(main())
