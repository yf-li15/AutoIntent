import os
from os.path import join
import time
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

dataname = args.dataset
gpu_id = args.gpu_id
ROOT_PATH = "./"   # add your root path
CODE_PATH = join(ROOT_PATH, 'code')

DATA_PATH = join(ROOT_PATH, 'data/'+dataname)

FILE_PATH = join(CODE_PATH, 'checkpoints')

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

# construct config
config = {}
config['batch_size'] = args.train_batch_size
config['test_batch_size'] = args.test_batch_size
config['emb_size'] = args.emb_size
config['layer']= args.layer
config['use_drop'] = args.use_drop
config['droprate']  = args.droprate
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['gamma'] = args.gamma
config['pretrain'] = args.pretrain
use_trans = args.use_trans
use_acf = args.use_acf
earlystop = args.earlystop

# add the datset details here
if dataname == 'Beijing':
    config['num_users'] = 11889
    config['num_locations'] = 13
    config['num_times'] = 96
    config['num_activities'] = 408
elif dataname == 'Shanghai':
    config['num_users'] = 10302
    config['num_locations'] = 13
    config['num_times'] = 96
    config['num_activities'] = 422


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed
modelname = args.model

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")