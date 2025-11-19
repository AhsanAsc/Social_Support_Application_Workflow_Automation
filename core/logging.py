import logging
import sys


def configure_logging() -> None:
    """
    Configure structured logging for the whole app.
    Call this once in FastAPI startup and before Streamlit if needed.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger("social_support_ai")
