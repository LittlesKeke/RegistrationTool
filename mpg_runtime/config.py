import os, torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
#device_id = [0,1,2]
DEVICE = torch.device(os.environ.get("REG_DEVICE", "cuda:1") if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
SEED = 666

## dataset
NUM_POINTS = 2048 # 降采样后的点数
NUM_TRANS = 100 # 随机变换数，生成该数字个随机变换后的点云用于学习
LOCAL_RATE = 0.7 # 局部比例
TRANSLATE = 0.5 # 初始平移 归一化坐标 0.5标准范围
ROTATE = 45.0 # 初始旋转 角度 45为标准范围
NUM_STAGES = 3 # 降采样阶段数
VOXEL_SIZE = 0.02 # 初始voxel大小


## train
ALPHA = 0.0 # 0表示il 2表示il+rl（rl主导）
log_name = 'M40'
PRETRAIN = False # 是否加载权重
RUN_MODE = 'train'  # 'train' 或 'eval'
BATCH_SIZE = 32 # 每回合batch大小
NUM_EPOCHS = 30 # 训练回合数
ITER_TRAIN, ITER_EVAL = 10, 10
NUM_TRAJ = 5 # 经验池数量
CLIP_VALUE = True
CLIP_EPS = 0.2
C_VALUE, C_ENTROPY = 0.3, 1e-2
LEARNING_RATE = 1e-4 if ALPHA == 0 else 1e-5 # 起始学习率，每10个epoch减半
log = True if log_name is not None else False # 是否开启log记录
MODEL_SAVE_PATH = './trained_models'


## model
IN_CHANNELS = 3
NUM_POINTS_IN_PATCH = 64
MATCHING_RADIUS = 0.05

BENCHMARK = False
FEAT_DIM = 1024
STATE_DIM = FEAT_DIM * 2
HEAD_DIM = 256

ACTION_DIM = 6  # 每个动作向量包含6维
NUM_ACTIONS = 6  # 6种动作 [+-x, +-y, +-z]
NUM_STEPSIZES = 6  # 6种步长
NUM_NOPS = 3  # 1种无操作的可能性


## buffer
GAMMA = 0.99
GAE_LAMBDA = 0.95
