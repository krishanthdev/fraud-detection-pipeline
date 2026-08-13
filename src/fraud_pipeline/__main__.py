"""Allow ``python -m fraud_pipeline`` as well as the installed ``fraud`` command."""

from fraud_pipeline.cli import app

if __name__ == "__main__":
    app()
