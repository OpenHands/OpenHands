"""Take screenshots of the OpenHands UI settings page via Playwright."""

import asyncio
import os

os.environ['PLAYWRIGHT_BROWSERS_PATH'] = os.path.expanduser('~/.cache/playwright')

from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1280, 'height': 900})

        # Navigate to the app settings page
        await page.goto('http://127.0.0.1:3000/app-settings', wait_until='networkidle')
        await page.wait_for_timeout(3000)

        await page.screenshot(
            path='/tmp/openhands-pr15103/.pr/logs/screenshot_settings.png',
            full_page=True,
        )
        print('Saved screenshot_settings.png')

        # Try to find and click on the MCP settings tab
        try:
            # Look for MCP-related elements
            mcp_tab = page.locator('text=MCP').first
            if await mcp_tab.count() > 0:
                await mcp_tab.click()
                await page.wait_for_timeout(2000)
                await page.screenshot(
                    path='/tmp/openhands-pr15103/.pr/logs/screenshot_mcp_tab.png',
                    full_page=True,
                )
                print('Saved screenshot_mcp_tab.png')
        except Exception as e:
            print(f'MCP tab click failed: {e}')

        # Take a screenshot of the full page after interactions
        await page.screenshot(
            path='/tmp/openhands-pr15103/.pr/logs/screenshot_full.png',
            full_page=True,
        )
        print('Saved screenshot_full.png')

        await browser.close()


asyncio.run(main())
