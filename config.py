import os

PATH = "./data"

# DATA files
TRAC1_FILE = os.path.join(PATH, "trac_data.csv")
HS_FILE = os.path.join(PATH, "hs_data.csv")
HOT_FILE = os.path.join(PATH, "hot_data.csv")

# Compute train and warmup steps from batch size
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 10.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 128

# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

SIMILARITY_THRESHOLD = 0.70

# Model
mBERT_MODULE_URL = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"
MuRIL_MODULE_URL = "https://tfhub.dev/google/MuRIL/1"
