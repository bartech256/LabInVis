class Config:
    """
    Configuration class for the whole pipeline.
    It contains parameters for data loading, graph construction,
    model training, and logging.
    It is designed to be easily extensible for different datasets,
    models, and training configurations.
    """
    def __init__(self):