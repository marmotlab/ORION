TEST_N_AGENTS = 3

INPUT_DIM = 9
EMBEDDING_DIM = 128
MAX_EPISODE_STEP = 128
UNBOUND_SPEED = False  # evader speed

USE_GPU = False
NUM_GPU = 0
NUM_META_AGENT = 30  # the number of processes
FOLDER_NAME = '09_30_0.051'
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/gifs'
fail_tests_path = f'fail_tests/{FOLDER_NAME}'

NUM_TEST = 50
SAVE_GIFS = False
SAVE_CSV = False