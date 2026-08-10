from pathlib import Path
import sys
from loguru import logger


def setup_logger(
    log_path: Path | None = None,
    stderr_verbosity="INFO",
    file_verbosity="DEBUG",
) -> int | None:
    """Setups the global logger with two sinks: a file and stderr.
    Returns a handle to the file sink.
    """

    logger.remove()
    logger.add(
        sys.stderr,
        level=stderr_verbosity,
        format="<level>{message}</level>",
        backtrace=False,
        diagnose=False,
    )

    if log_path is not None:
        logger.info(f"Logging in {log_path}")
        log_path.parent.mkdir(exist_ok=True, parents=True)
        if log_path.exists():
            logger.warning(f"Overwriting logfile: {log_path}")
            log_path.unlink()
        return logger.add(
            log_path,
            level=file_verbosity,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{line}: in {function}:</cyan> <level>{message}</level>",
        )
