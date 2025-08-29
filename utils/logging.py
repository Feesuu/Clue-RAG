import os
import logging
import sys
from datetime import datetime
from typing import Optional
from tqdm import tqdm
from utils.config import BaseConfig

def get_logger(
    config: Optional[BaseConfig] = None,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Initialize and configure a logger that creates new log files for each run.
    
    Args:
        config: BaseConfig instance (for default save directory)
        log_dir: Custom log directory (optional, overrides config.save_dir)
        log_file: Custom log filename (optional)
        level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logging.Logger instance
    """
    # 1. Determine log directory
    if log_dir is None:
        if config is not None and hasattr(config, 'save_dir'):
            
            log_dir = os.path.join(os.path.join(config.save_dir, f"{config.select_metric}_{config.alpha:.2f}"), "logs")
        else:
            log_dir = os.getenv("LOG_DIR", os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                "logs"))

    os.makedirs(log_dir, exist_ok=True)

    # 2. Generate timestamped log filename
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"exp_{timestamp}.log"
    log_path = os.path.join(log_dir, log_file)

    # 3. Create logger with fixed name "experiment"
    logger = logging.getLogger("experiment")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # 4. Configure log format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 5. Add file handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. Add stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

config = BaseConfig()
logger = get_logger(config=config, level=logging.INFO)