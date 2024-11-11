import logging

# ANSI color codes for different log levels
COLOR_CODES = {
    "DEBUG": "\033[94m",     # Blue
    "INFO": "\033[92m",      # Green
    "WARNING": "\033[93m",   # Yellow
    "ERROR": "\033[91m",     # Red
    "CRITICAL": "\033[95m",  # Magenta
    "RESET": "\033[0m"       # Reset to default color
}

class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Set color based on log level
        log_color = COLOR_CODES.get(record.levelname, COLOR_CODES["RESET"])
        reset_color = COLOR_CODES["RESET"]

        # Define log format with colors
        log_format = f"{log_color}%(levelname)s: %(message)s{reset_color}"

        # Set formatter to use colorized format
        formatter = logging.Formatter(log_format)
        return formatter.format(record)

def get_logger() -> logging.Logger:
    """
    Create a colorized dataflow logger instance.
    :return: a logger instance
    """
    logger = logging.getLogger('BookDataPipeline')
    logger.setLevel(logging.INFO)

    # Create console handler with colorized formatter
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter())

    # Add handler to logger
    if not logger.handlers:  # Avoid adding multiple handlers
        logger.addHandler(ch)
    
    return logger
