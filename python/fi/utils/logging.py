import logging
import sys


class CustomLogFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[33m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.blue + self.fmt + self.reset,
            logging.INFO: self.grey + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.propagate = False
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.INFO)
fmt = "  %(name)s | %(levelname)s | %(message)s"
if hasattr(sys, "ps1"):  # for python interactive mode
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(CustomLogFormatter(fmt))
    logger.addHandler(handler)


def get_truncation_warning_message(instance, limit) -> str:
    return (
        f"Attention: {instance} exceeding the {limit} character limit will be "
        "automatically truncated upon ingestion into the fi platform. Should you require "
        "a higher limit, please reach out to our support team at support@fi.com"
    )
