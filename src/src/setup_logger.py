import logging.config
from src.src.my_confs.conf_data_engineering import fname

logging.config.fileConfig(
    fname=fname,
    disable_existing_loggers=False,
)

logger = logging.getLogger(__name__)
