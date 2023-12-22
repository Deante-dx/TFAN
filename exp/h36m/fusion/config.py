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

"""please config ROOT_dir and user when u first using"""
C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'TFAN'
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]


C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))
C.exp_name = osp.abspath(osp.join(C.log_dir, "acc_log"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'




"""Dataset Config"""
C.h36m_anno_dir = osp.join(C.root_dir, 'data/h36m/')
C.motion = edict()

C.motion.h36m_input_length = 50
C.motion.h36m_input_length_dct = 50
C.motion.h36m_target_length_train = 10
C.motion.h36m_target_length_eval = 25
C.motion.dim = 66

C.data_aug = True

"""Train Config"""
C.batch_size = 256
C.num_workers = 8

C.cos_lr_max = 1e-5
C.cos_lr_min = 5e-8
C.cos_lr_total_iters = 2000

C.weight_decay = 1e-4
C.model_pth = None

"""Eval Config"""
C.shift_step = 1

"""Display Config"""
C.print_every = 100
C.save_every = 500


if __name__ == '__main__':
    print(config.decoder.motion_mlp)
