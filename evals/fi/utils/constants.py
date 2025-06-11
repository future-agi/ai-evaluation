import os

MIN_PREDICTION_ID_LEN = 1
MAX_PREDICTION_ID_LEN = 128

# The maximum number of character for tag values
MAX_TAG_LENGTH = 20_000
MAX_TAG_LENGTH_TRUNCATION = 1_000


MAX_PAST_YEARS_FROM_CURRENT_TIME = 5
MAX_FUTURE_YEARS_FROM_CURRENT_TIME = 1

MAX_RAW_DATA_CHARACTERS = 50_000
MAX_RAW_DATA_CHARACTERS_TRUNCATION = 5_000

# The maximum number of embeddings
MAX_NUMBER_OF_EMBEDDINGS = 30
MAX_EMBEDDING_DIMENSIONALITY = 20_000

RESERVED_TAG_COLS = []

# Authentication via environment variables
SECRET_KEY_ENVVAR_NAME = "FI_SECRET_KEY"
API_KEY_ENVVAR_NAME = "FI_API_KEY"

def get_base_url():
    """Get the base URL from environment variable at runtime.
    
    This ensures that changes to the FI_BASE_URL environment variable
    are picked up even after the module has been imported.
    
    Returns:
        str: The base URL for the FutureAGI API
    """
    return os.getenv("FI_BASE_URL", "https://api.futureagi.com")

# Session settings
DEFAULT_TIMEOUT = 200
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_QUEUE = 5000


# Dataset settings
PAGE_SIZE = 100
DATASET_TEMP_FILE_PREFIX = "tmp_fi_dataset_"
DATASET_TEMP_FILE_SUFFIX = ".csv"

# Environment variables specific to the subpackage
FI_PROJECT_NAME = "DEFAULT_PROJECT_NAME"
FI_PROJECT_VERSION_NAME = "DEFAULT_PROJECT_VERSION_NAME"
