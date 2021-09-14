import tensorflow as tf
from easydict import EasyDict as edict


__C = edict()
cfg = __C


# MODELS
__C.ENC = edict()
__C.ENC.VOCAB_SIZE = 6913
__C.ENC.MAX_SEQ_LEN = 30
__C.ENC.SEQ_DIM = 200
__C.ENC.FAN_MID = 64
__C.ENC.NUM_PROC_BLOCKS = 2
__C.ENC.NUM_ATTN_HEADS = 4
__C.ENC.NUM_FINAL_CHANNELS = 128

__C.GEN = edict()
__C.GEN.LATENT_DIM = 128
__C.GEN.LATENT_SEQ_DIM = 128
__C.GEN.NUM_CHANNELS = 3
__C.GEN.OPT_B1 = 0.0
__C.GEN.OPT_B2 = 0.99
__C.GEN.OPT_EPS = 1e-8

__C.CRITIC = edict()
__C.CRITIC.NUM_CHANNELS = 3
__C.CRITIC.GROUP_DIM = 4
__C.CRITIC.LABEL_DIM = 0
__C.CRITIC.OPT_B1 = 0.0
__C.CRITIC.OPT_B2 = 0.99
__C.CRITIC.OPT_EPS = 1e-8


# TRAINING
__C.TRAIN = edict()
__C.TRAIN.DTYPE = tf.float32
__C.TRAIN.FORMAT = "NHWC"
__C.TRAIN.AUG_RES_FC = 1.15

__C.TRAIN.NUM_SAMPLES = 11787
__C.TRAIN.GP_LAMBDA = 5.0
__C.TRAIN.MA_GP_LAMBDA = 2.0
__C.TRAIN.MA_GP_EXP = 6.0
__C.TRAIN.CRITIC_ITERS = 1
__C.TRAIN.CHECKPOINT_PERIOD = 1000
__C.TRAIN.SAMPLE_STATUS_PERIOD = 20

__C.TRAIN.RES_LIST = ["4", "8", "16", "32", "64", "128"]
__C.TRAIN.EPOCHS_PER_RES = {"4": 40000, "8": 100000, "16": 150000, "32": 300000, "64": 300000, "128": 1000000} # {"4": 3, "8": 3, "16": 3, "32": 3, "64": 3, "128": 3} #
__C.TRAIN.BATCH_SIZE_PER_RES = {"4": 128, "8": 64, "16": 32, "32": 16, "64": 16, "128": 8}
__C.TRAIN.LR = {"4": 0.001, "8": 0.001, "16": 0.001, "32": 0.001, "64": 0.001, "128": 0.001}


# DIRECTORIES
__C.DIRS = edict()
__C.DIRS.RAW_DATA_DIR = "data_raw/CUB"
__C.DIRS.PROC_DATA_DIR = "data_processed/CUB"
__C.DIRS.RESULTS_DIR = "results"
__C.DIRS.CHECKPOINTS_DIR = "checkpoints"
__C.DIRS.CHECKPOINTS_GDRIVE_DIR = "checkpoints"
__C.DIRS.TRAIN_LOGS_DIR = "training_logs"
__C.DIRS.RAW_IMG_DIR = f"{__C.DIRS.RAW_DATA_DIR}/images"
__C.DIRS.RAW_STYLE_DIR = f"data_raw/style"
__C.DIRS.RAW_TXT_DIR = f"{__C.DIRS.RAW_DATA_DIR}/text"
__C.DIRS.PROC_TXT_DIR = f"{__C.DIRS.PROC_DATA_DIR}/captions"
__C.DIRS.PROC_IMG_DIRS = [f"{__C.DIRS.PROC_DATA_DIR}/{res}x{res}" for res in __C.TRAIN.RES_LIST]
__C.DIRS.PROC_STYLE_DIRS = [f"data_processed/style/{res}x{res}" for res in __C.TRAIN.RES_LIST]
__C.DIRS.SAMPLE_BATCH_DIR = [f"{__C.DIRS.PROC_DATA_DIR}/sample_batch_viz/{res}x{res}" for res in __C.TRAIN.RES_LIST]

