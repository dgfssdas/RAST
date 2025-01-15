import torch
import numpy as np
import random
from utils.log_helper import *


def set_log(args):
    log_config(args=args, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info('=' * 30)
    logging.info(' ' * 9 + 'Dataset {}'.format(args.dataset) + ' ' * 9)
    logging.info('=' * 30)
    args_dict = args.__dict__
    parameters = 'Hyper-parameters: \n'
    for idx, (key, value) in enumerate(args_dict.items()):
        if idx == len(args_dict) - 1:
            parameters += '\t\t{}: {}'.format(key, str(value))
        else:
            parameters += '\t\t{}: {}\n'.format(key, str(value))
    logging.info(parameters)
    logging.info('=' * 32)


def set_seed(seed_num):
    seed = seed_num
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def get_total_parameters(model):
    return str(sum(p.numel() for p in model.parameters() if p.requires_grad))
