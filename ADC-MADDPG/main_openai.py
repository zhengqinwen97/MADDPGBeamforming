import torch
import random
import sys
import os

from function import *
from arguments import parse_args


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
if __name__ == '__main__':
    arglist = parse_args()
    set_seed(42)
    train_mix_ax(arglist, type="no_jammer")


