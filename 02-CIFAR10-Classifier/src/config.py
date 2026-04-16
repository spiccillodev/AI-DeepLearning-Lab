import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "../data")
MODEL_PATH = os.path.join(BASE_DIR, "../models/modello_cifar10.pt")

LOG_DIR = os.path.join(BASE_DIR, "../outputs/logs")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "../outputs/predictions")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "../test/img")