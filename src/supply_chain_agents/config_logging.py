import logging


def get_logger(obj) -> logging.Logger:
    """Return a logger named from a module name or object's class"""
    if isinstance(obj, str):
        return logging.getLogger(obj)
    return logging.getLogger(f"{obj.__class__.__module__}.{obj.__class__.__name__}")


def configure_logging() -> None:  # pragma: no cover
    """Configure application-wide logging format and default level"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
