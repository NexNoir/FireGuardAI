from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOG_DIR / "fireguard.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"


def setup_logging():
    """
    Configure FireGuard logging safely.

    This function can be called multiple times without adding
    duplicate handlers.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("fireguard")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


def get_logger(name=None):
    """
    Return a configured FireGuard logger.

    Secrets such as API keys must never be passed directly
    to log messages.
    """

    setup_logging()

    if name:
        return logging.getLogger(f"fireguard.{name}")

    return logging.getLogger("fireguard")


if __name__ == "__main__":
    logger = get_logger("logging_test")

    logger.info("FireGuard logging test started")
    logger.warning("FireGuard logging warning test")
    logger.error("FireGuard logging error test")

    print("=" * 70)
    print("FireGuard — Stage 14 Logging Test")
    print("=" * 70)
    print(f"Log directory : {LOG_DIR}")
    print(f"Main log      : {LOG_FILE}")
    print(f"Error log     : {ERROR_LOG_FILE}")
    print("STATUS: LOGGING READY")