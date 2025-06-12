"""
HistData.com forex data downloader using Playwright.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional, List

from playwright.async_api import async_playwright


logger = logging.getLogger(__name__)


class HistDataDownloader:
    """Downloads historical forex data from HistData.com using Playwright."""

    def __init__(self, download_dir: Optional[str] = None):
        """
        Initialize the downloader.
        
        Args:
            download_dir: Directory to save downloaded files. If None, uses current directory.
        """
        self.download_dir = Path(download_dir) if download_dir else Path.cwd()
        self.download_dir.mkdir(parents=True, exist_ok=True)

    async def download_forex_data(
        self,
        currency_pair: str,
        year: int,
        timeframe: str = "1-minute-bar-quotes",
        headless: bool = True
    ) -> Optional[Path]:
        """
        Download forex data for a specific currency pair and year.
        
        Args:
            currency_pair: Currency pair (e.g., 'usdjpy', 'eurusd')
            year: Year to download data for
            timeframe: Timeframe for the data (default: '1-minute-bar-quotes')
            headless: Whether to run browser in headless mode
            
        Returns:
            Path to the downloaded file, or None if download failed
        """
        url = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/{timeframe}/{currency_pair.lower()}/{year}"
        logger.info(f"Downloading {currency_pair} {year} data from {url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            
            # Navigate to the download page
            await page.goto(url)
            
            # Look for the download link
            download_link_pattern = f"HISTDATA_COM_ASCII_{currency_pair.upper()}_M1_{year}.zip"
            logger.info(f"Looking for download link: {download_link_pattern}")

            # Find the download link
            download_link = page.get_by_role("link", name=download_link_pattern)
            
            # Set up download path
            expected_filename = download_link_pattern
            file_path = self.download_dir / expected_filename
            
            # Start download
            async with page.expect_download() as download_info:
                await download_link.click()
            
            download = await download_info.value
            
            # Save the downloaded file
            await download.save_as(file_path)
            logger.info(f"Downloaded file saved to: {file_path}")
            
            await browser.close()
            return file_path
        