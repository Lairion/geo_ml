import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BASE_DIR = Path(__file__).parent

ROOT_DIR = BASE_DIR.parent

DATA_DIR = ROOT_DIR.joinpath(
    'data', 'satelit'
)
CHECK_PICTURE = ROOT_DIR.joinpath(
    "data", "check_picture", "t_cross", "picture.PNG"
)
MAP_DIR = ROOT_DIR.joinpath(
    'data', 'map'
)
MODELS_DIR = BASE_DIR.joinpath("models")

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
EPOCHS = 30
