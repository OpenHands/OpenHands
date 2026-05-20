"""Playwright browser helper utilities for BDD tests.

Provides convenience functions for browser automation including navigation,
element interaction, text assertions, and screenshots.

Usage:
    await wait_for_element(page, ".chat-input", timeout=5)
    await fill_input(page, ".chat-input", "hello")
    await click_button(page, "Send")
"""

from __future__ import annotations

from playwright.async_api import Page


async def wait_for_element(page: Page, selector: str, timeout: float = 5.0) -> None:
    """Wait for element to appear on page.

    Args:
        page: Playwright page object
        selector: CSS selector
        timeout: Timeout in seconds

    Raises:
        TimeoutError: If element doesn't appear within timeout
    """
    await page.wait_for_selector(selector, timeout=timeout * 1000)


async def wait_for_text(page: Page, text: str, timeout: float = 5.0) -> None:
    """Wait for text to appear on page.

    Args:
        page: Playwright page object
        text: Text to wait for
        timeout: Timeout in seconds

    Raises:
        TimeoutError: If text doesn't appear within timeout
    """
    await page.wait_for_function(
        f"() => document.body.innerText.includes('{text}')",
        timeout=timeout * 1000,
    )


async def wait_for_text_gone(page: Page, text: str, timeout: float = 5.0) -> None:
    """Wait for text to disappear from page.

    Args:
        page: Playwright page object
        text: Text to wait for
        timeout: Timeout in seconds

    Raises:
        TimeoutError: If text is still present after timeout
    """
    await page.wait_for_function(
        f"() => !document.body.innerText.includes('{text}')",
        timeout=timeout * 1000,
    )


async def fill_input(page: Page, selector: str, text: str) -> None:
    """Fill input field with text.

    Args:
        page: Playwright page object
        selector: CSS selector for input
        text: Text to fill
    """
    await page.fill(selector, text)


async def clear_input(page: Page, selector: str) -> None:
    """Clear input field.

    Args:
        page: Playwright page object
        selector: CSS selector for input
    """
    await page.fill(selector, '')


async def click_button(page: Page, text_or_selector: str) -> None:
    """Click button by text or selector.

    Args:
        page: Playwright page object
        text_or_selector: Button text (e.g., "Send") or CSS selector
    """
    if text_or_selector.startswith('.') or text_or_selector.startswith('#'):
        # CSS selector
        await page.click(text_or_selector)
    else:
        # Button text
        await page.click(f"button:has-text('{text_or_selector}')")


async def click_element(page: Page, selector: str) -> None:
    """Click element by selector.

    Args:
        page: Playwright page object
        selector: CSS selector
    """
    await page.click(selector)


async def get_text(page: Page, selector: str) -> str:
    """Get text content of element.

    Args:
        page: Playwright page object
        selector: CSS selector

    Returns:
        Text content
    """
    return await page.text_content(selector) or ''


async def get_input_value(page: Page, selector: str) -> str:
    """Get value of input field.

    Args:
        page: Playwright page object
        selector: CSS selector

    Returns:
        Input value
    """
    return await page.input_value(selector)


async def is_visible(page: Page, selector: str) -> bool:
    """Check if element is visible.

    Args:
        page: Playwright page object
        selector: CSS selector

    Returns:
        True if visible
    """
    try:
        await wait_for_element(page, selector, timeout=0.5)
        return True
    except Exception:
        return False


async def is_hidden(page: Page, selector: str) -> bool:
    """Check if element is hidden.

    Args:
        page: Playwright page object
        selector: CSS selector

    Returns:
        True if hidden
    """
    return not await is_visible(page, selector)


async def press_key(page: Page, key: str) -> None:
    """Press keyboard key.

    Args:
        page: Playwright page object
        key: Key name (e.g., 'Enter', 'Escape')
    """
    await page.press('body', key)


async def focus_element(page: Page, selector: str) -> None:
    """Focus element.

    Args:
        page: Playwright page object
        selector: CSS selector
    """
    await page.focus(selector)


async def get_attribute(page: Page, selector: str, attribute: str) -> str | None:
    """Get element attribute value.

    Args:
        page: Playwright page object
        selector: CSS selector
        attribute: Attribute name

    Returns:
        Attribute value or None
    """
    return await page.get_attribute(selector, attribute)


async def count_elements(page: Page, selector: str) -> int:
    """Count elements matching selector.

    Args:
        page: Playwright page object
        selector: CSS selector

    Returns:
        Number of matching elements
    """
    return await page.query_selector_all(selector).__len__()


async def take_screenshot(page: Page, name: str) -> None:
    """Take screenshot of page.

    Args:
        page: Playwright page object
        name: Screenshot filename (without extension)
    """
    await page.screenshot(path=f'/tmp/{name}.png')


async def scroll_to_bottom(page: Page) -> None:
    """Scroll page to bottom.

    Args:
        page: Playwright page object
    """
    await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')


async def scroll_to_top(page: Page) -> None:
    """Scroll page to top.

    Args:
        page: Playwright page object
    """
    await page.evaluate('window.scrollTo(0, 0)')


async def reload_page(page: Page) -> None:
    """Reload page.

    Args:
        page: Playwright page object
    """
    await page.reload()


async def go_back(page: Page) -> None:
    """Navigate back in browser history.

    Args:
        page: Playwright page object
    """
    await page.go_back()


async def go_forward(page: Page) -> None:
    """Navigate forward in browser history.

    Args:
        page: Playwright page object
    """
    await page.go_forward()


async def get_page_title(page: Page) -> str:
    """Get page title.

    Args:
        page: Playwright page object

    Returns:
        Page title
    """
    return await page.title()


async def get_page_url(page: Page) -> str:
    """Get current page URL.

    Args:
        page: Playwright page object

    Returns:
        Current URL
    """
    return page.url
