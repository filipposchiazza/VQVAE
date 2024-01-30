# Configuration parameters
DEVICE = 'cpu' 
SAVE_FOLDER = './models/01'

# Dataset parameters
IMG_DIR = 'images/'
FRACTION = 1.0 # Fraction of the dataset to use
TRANSFORM = None # Transform to apply to the dataset

# Model parameters
INPUT_CHANNELS = 3 # 3 for RGB, 1 for grayscale
OUT_CHANNELS = 3 # 3 for RGB, 1 for grayscale
CHANNELS = [64, 128, 256, 512]
NUM_RES_BLOCK = 2
NUM_EMB_VECTORS = 256
EMB_DIM = 512
GROUPS = 16
BETA = 0.25
EMA_UPDATE = True # Whether to update with EMA the codebooks

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1

# Early stopping parameters
PATIENCE = 10
MIN_DELTA = 0.0
