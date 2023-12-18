import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mse, masked_mae

from .arch import Corrformer

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "Corrformer model configuration"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "Electricity"
CFG.DATASET_TYPE = "Electricity Consumption"
CFG.DATASET_INPUT_LEN = 336
CFG.DATASET_OUTPUT_LEN = 336
CFG.GPU_NUM = 1
# CFG.RESCALE = False

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "Corrformer"
CFG.MODEL.ARCH = Corrformer
NUM_NODES = 321
NUM_Variable = 107
CFG.MODEL.PARAM = EasyDict(
    {
    "enc_in": NUM_Variable//NUM_NODES,
    "dec_in": NUM_Variable//NUM_NODES,
    "c_out":  NUM_Variable//NUM_NODES,
    "seq_len": CFG.DATASET_INPUT_LEN,
    "label_len": CFG.DATASET_INPUT_LEN//2,       # start token length used in decoder
    "pred_len": CFG.DATASET_OUTPUT_LEN,         # prediction sequence length
    "factor_temporal": 1,                       # attn factor
    "factor_spatial": 1,                        # attn factor
    "d_model": 16,
    "moving_avg": 25,                           # window size of moving average. This is a CRUCIAL hyper-parameter.
    "n_heads": 8,
    "e_layers": 2,                              # num of encoder layers
    "d_layers": 1,                              # num of decoder layers
    "d_ff": 32,
    "dropout": 0.05,
    "variable_num": NUM_Variable,
    "node_num":  NUM_NODES,                   # num nodes
    "node_list": [1,107],
    "enc_tcn_layers":1,
    "dec_tcn_layers":1,
    "output_attention": False,
    "embed": "timeF",                           # [timeF, fixed, learned]
    "activation": "gelu",
    "num_time_features": 4,                     # number of used time features
    "time_of_day_size": 24,
    "day_of_week_size": 7,
    "day_of_month_size": 31,
    "day_of_year_size": 366
    }
)
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0002,
    "weight_decay": 0.0001,
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 50
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 16
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [12, 24, 48, 96, 192, 288, 336]

