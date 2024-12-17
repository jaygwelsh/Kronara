# kronara/utils/logging_utils.py
from loguru import logger
import sys

def get_logger():
    logger.remove()
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    return logger
