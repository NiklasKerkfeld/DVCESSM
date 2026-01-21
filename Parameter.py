import os
from dotenv import load_dotenv

load_dotenv()

# dataset properties
N_CLASSES = int(os.getenv('N_CLASSES'))
IN_CHANNELS = int(os.getenv('IN_CHANNELS'))
IN_HEIGHT = int(os.getenv('IN_HEIGHT'))
IN_WIDTH = int(os.getenv('IN_WIDTH'))

# trainings properties
ITERATIONS = int(os.getenv('ITERATIONS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))

# Classifier parameters
STD_CLS = float(os.getenv('STD_CLS'))
P_CLS = float(os.getenv('P_CLS'))
B_CLS = float(os.getenv('B_CLS'))
B_H_MAX_CLS = int(os.getenv('B_H_MAX_CLS'))
B_W_MAX_CLS = int(os.getenv('B_W_MAX_CLS'))

# Generator parameters
ALPHA = float(os.getenv('ALPHA'))
BETA = float(os.getenv('BETA'))
LR_GEN = float(os.getenv('LR_GEN'))
LR_DISC = float(os.getenv('LR_DISC'))
WD_DISC = float(os.getenv('WD_DISC'))
L1_REG = float(os.getenv('L1_REG'))
L2_REG = float(os.getenv('L2_REG'))

# Discriminator parameters
STD_DISC = float(os.getenv('STD_DISC'))
P_DISC = float(os.getenv('P_DISC'))
B_DISC = float(os.getenv('B_DISC'))
B_H_MAX_DISC = int(os.getenv('B_H_MAX_DISC'))
B_W_MAX_DICS = int(os.getenv('B_W_MAX_DICS'))
LABEL_SMOOTHING = os.getenv('LABEL_SMOOTHING') == 'True'

# Diffusion parameter
T = int(os.getenv('T'))
BETA_MIN = float(os.getenv('BETA_MIN'))
BETA_MAX = float(os.getenv('BETA_MAX'))
TIME_DIM = int(os.getenv('TIME_DIM'))