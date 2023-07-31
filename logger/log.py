import logging
import colorlog

__logger = logging.getLogger('my_logger')
__logger.setLevel(logging.DEBUG)

__handler = colorlog.StreamHandler()
__formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
)
__handler.setFormatter(__formatter)
__logger.addHandler(__handler)


def log_debug(message):
    __logger.debug(message)


def log_info(message):
    __logger.info(message)


def log_warning(message):
    __logger.warning(message)


def log_error(message):
    __logger.error(message)


def log_critical(message):
    __logger.critical(message)
