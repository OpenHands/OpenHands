from server.auth.sheets_client import GoogleSheetsClient

from openhands.app_server.utils.logger import openhands_logger


def test_import():
    assert openhands_logger is not None
    assert GoogleSheetsClient is not None
