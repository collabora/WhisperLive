"""
Reliable webhook delivery with exponential backoff retry.

Used for async transcription callback URLs. Retries failed deliveries
with configurable backoff and max attempts.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def send_webhook(
    url: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: int = 30,
) -> bool:
    """Send a webhook POST request with retry and exponential backoff.

    Args:
        url: The callback URL.
        payload: JSON-serializable dict to send.
        max_retries: Maximum number of retry attempts (0 = no retries).
        initial_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiply delay by this factor after each retry.
        timeout: HTTP request timeout in seconds.

    Returns:
        True if delivery succeeded, False if all attempts failed.
    """
    data = json.dumps(payload).encode("utf-8")
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urllib.request.urlopen(req, timeout=timeout)
            status = response.getcode()

            if 200 <= status < 300:
                logger.info(f"Webhook delivered to {url} (attempt {attempt + 1}, status {status})")
                return True
            else:
                logger.warning(f"Webhook to {url} returned {status} (attempt {attempt + 1})")

        except urllib.error.HTTPError as e:
            status = e.code
            # Don't retry 4xx client errors (except 429 rate limit)
            if 400 <= status < 500 and status != 429:
                logger.error(f"Webhook to {url} failed with {status} (not retrying)")
                return False
            logger.warning(f"Webhook to {url} failed with {status} (attempt {attempt + 1})")

        except urllib.error.URLError as e:
            logger.warning(f"Webhook to {url} connection failed: {e.reason} (attempt {attempt + 1})")

        except Exception as e:
            logger.warning(f"Webhook to {url} error: {e} (attempt {attempt + 1})")

        # Wait before retry (skip wait after last attempt)
        if attempt < max_retries:
            logger.info(f"Retrying webhook to {url} in {delay:.1f}s...")
            time.sleep(delay)
            delay *= backoff_factor

    logger.error(f"Webhook to {url} failed after {max_retries + 1} attempts")
    return False
