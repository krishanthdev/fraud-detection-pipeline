"""End to end credit card fraud detection pipeline.

The package is organised as one module per pipeline stage:

    ingest      stage 1, read the raw file and write a typed interim table
    validate    stage 2, enforce schema and data quality rules
    features    stage 3, build model ready features without leaking the future
    train       stage 4, fit every model under every imbalance strategy
    evaluate    stage 5, score models, tune the threshold, explain predictions
    registry    stage 6, log runs to MLflow and promote a champion
    serve       stage 7, the FastAPI scoring service
    package     stage 8, the Docker build

Every stage reads ``configs/config.yaml`` and writes into ``data/`` or ``models/``.
"""

__version__ = "0.1.0"
