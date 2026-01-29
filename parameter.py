# saving path
FOLDER_NAME = 'orion'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
fail_examples_path = f'fail_examples/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 64  # how many training steps before writing data to tensorboard
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 400  # how many episodes before saving a gifs
MIN_CENTERS_BEFORE_SPARSIFY = 25
SPARSIFICATION_CENTERS_KNN_RAD = 20
CENTER_SIZE = 25
CERTAIN_CONFIDENCE = 1
UNCERTAIN_CONFIDENCE = 0
CERTAIN = 1
UNCERTAIN = 0
NUM_SIM_STEPS = 10

# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 4.0  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers
N_TARGET = 3
N_AGENTS = 3
GROUP_START = True

# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value used in belief mask, represent the pixel out of the sensor range
FALSE_POSITIVE = 210   # need further verification, now is free space
FALSE_NEGATIVE = 90   # need further verification, now is obstacle

# sensor and utility range
SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 2  # ignore the utility if observable frontiers are less than this value

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 5000
BATCH_SIZE = 64
LR = 1e-5
GAMMA = 0.99
NUM_META_AGENT = 30  # how many threads does your CPU have

# network parameters
NODE_INPUT_DIM = 9
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes, fixed
NODE_PADDING_SIZE = 600  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 0  # 0 unless you want to collect data using GPUs

