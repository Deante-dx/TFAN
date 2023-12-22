from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import time
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 888
C.test = 'baseline.txt'

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'TFAN'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]


C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))


exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'


C.h36m_anno_dir = osp.join(C.root_dir, 'data/cmu_mocap/')
C.motion = edict()

C.motion.cmu_input_length = 50
C.motion.cmu_input_length_dct = 50
C.motion.cmu_target_length_train = 10
C.motion.cmu_target_length_eval = 25
C.motion.dim = 75


"""Train Config"""
C.batch_size = 64

C.cos_lr_max=3e-4
C.cos_lr_min=5e-8
C.cos_lr_total_iters=50000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 5

"""Display Config"""
C.print_every = 100
C.save_every = 5000


if __name__ == '__main__':
    print(config.decoder.motion_mlp)
