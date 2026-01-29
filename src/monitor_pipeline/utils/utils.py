from pathlib import Path
import logging
import argparse
import os


def mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_logger(logger_name, logger_path, logger_format):
    logger = logging.getLogger(logger_name)

    if not logger.handlers:  # Check if the logger has no handlers yet
        # Configure the root logger
        logging.basicConfig(filename=logger_path, level=logging.INFO, format=logger_format)

        # Create a file handler
        handler = logging.FileHandler(logger_path)
        handler.setFormatter(logging.Formatter(logger_format))

        # Add the file handler to the logger
        logger.addHandler(handler)

        # Set propagate to False in order to avoid double entries
        logger.propagate = False

    return logger


def try_remove_tmpfiles(path):
    logging.info(f'Try clean-up tmpfiles')
    for file in os.listdir(path):
        if file.endswith(".nc"):
            try:
                os.remove(os.path.join(path, file))
                logging.info(f"Removed: {file}")
            except Exception as e:
                logging.info(f"Failed to remove {file}: {e}")
                

def get_sys_argv():
    parser = argparse.ArgumentParser(description="Parse required arguments for the analysis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-j", "--json-file", help="Point to json file that contains required parameters", required=True)

    args = parser.parse_args()
    config = vars(args)
    return config