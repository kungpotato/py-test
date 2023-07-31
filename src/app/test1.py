
import logging
import colorlog

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Create a ColorHandler with a colored output format
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
)
handler.setFormatter(formatter)
logger.addHandler(handler)

name = 'ghost busters'
weight = 65.5

logger.info(f"{name} => {weight}")
