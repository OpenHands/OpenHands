from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import gspread
from google.auth import default

from openhands.core.logger import openhands_logger as logger


class GoogleSheetsClient:
    def __init__(self):
        """Initialize Google Sheets client using workload identity.
        Uses application default credentials which supports workload identity when running in GCP.
        """
        logger.info('Initializing Google Sheets client with workload identity')
        self.client = None
        self._cache: Dict[Tuple[str, str], Tuple[List[str], datetime]] = {}
        self._cache_ttl = timedelta(seconds=15)
        try:
            credentials, project = default(
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            logger.info(f'Successfully obtained credentials for project: {project}')
            self.client = gspread.authorize(credentials)
            logger.info('Successfully initialized Google Sheets API service')
        except Exception:
            logger.exception('Failed to initialize Google Sheets client')
            self.client = None

    def _get_from_cache(
        self, spreadsheet_id: str, range_name: str
    ) -> Optional[List[str]]:
        """Get usernames from cache if available and not expired.

        Args:
            spreadsheet_id: The ID of the Google Sheet
            range_name: The A1 notation of the range to fetch
        Returns:
            List of usernames if cache hit and not expired, None otherwise
        """
        cache_key = (spreadsheet_id, range_name)
        if cache_key not in self._cache:
            return None

        usernames, timestamp = self._cache[cache_key]
        if datetime.now() - timestamp > self._cache_ttl:
            logger.info('Cache expired, will fetch fresh data')
            return None

        logger.info(
            f'Using cached data from {timestamp.isoformat()} '
            f'({len(usernames)} usernames)'
        )
        return usernames

    def _update_cache(
        self, spreadsheet_id: str, range_name: str, usernames: List[str]
    ) -> None:
        """Update cache with new usernames and current timestamp.

        Args:
            spreadsheet_id: The ID of the Google Sheet
            range_name: The A1 notation of the range to fetch
            usernames: List of usernames to cache
        """
        cache_key = (spreadsheet_id, range_name)
        self._cache[cache_key] = (usernames, datetime.now())

    def get_usernames(self, spreadsheet_id: str, range_name: str = 'A:A') -> List[str]:
        """Get list of usernames from specified Google Sheet.
        Uses cached data if available and less than 15 seconds old.

        Args:
            spreadsheet_id: The ID of the Google Sheet
            range_name: The A1 notation of the range to fetch
        Returns:
            List of usernames from the sheet
        """
        if not self.client:
            logger.error('Google Sheets client not initialized')
            return []

        # Try to get from cache first
        cached_usernames = self._get_from_cache(spreadsheet_id, range_name)
        if cached_usernames is not None:
            return cached_usernames

        try:
            logger.info(
                f'Fetching usernames from sheet {spreadsheet_id}, range {range_name}'
            )
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.sheet1  # Get first worksheet
            values = worksheet.get(range_name)

            usernames = [
                str(cell[0]).strip() for cell in values if cell and cell[0].strip()
            ]
            logger.info(
                f'Successfully fetched {len(usernames)} usernames from Google Sheet'
            )

            # Update cache with new data
            self._update_cache(spreadsheet_id, range_name, usernames)
            return usernames

        except gspread.exceptions.APIError:
            logger.exception(f'Error accessing Google Sheet {spreadsheet_id}')
            return []
        except Exception:
            logger.exception(
                f'Unexpected error accessing Google Sheet {spreadsheet_id}'
            )
            return []
