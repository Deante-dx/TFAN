import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
import copy

from config import config
from lib.datasets.h36m import H36MDataset
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import link_file, ensure_dir
from lib.datasets.h36m_eval import H36MEval
from models.Predictor import Predictor
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange
from models.FusionModel import FusionModel
from Fusion import *
from test import test

torch.use_deterministic_algorithms(True)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

acc_log = open(config.exp_name, 'a')
torch.manual_seed(config.seed)
writer = SummaryWriter()

acc_log.write(''.join('Seed : ' + str(config.seed) + '\n'))

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def update_lr_multistep(nb_iter, optimizer):
    if nb_iter > 1500:
        current_lr = 1e-5
    else:
        current_lr = 3e-4
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm


def get_bone_loss(gt, pred):
    pred = pred.reshape(pred.shape[0], pred.shape[1], 22, 3)
    pred = pred.reshape(-1, 3)
    gt = gt.reshape(-1, 3).cuda()
    loss = torch.mean(torch.norm(pred - gt, 2, 1))
    return loss


def get_bp_model():
    p_model_pth = r'E:\DX\MLPbaseline别动\siMLPe-原始\exps\baseline_h36m\log\position_best\model-iter-35000.pth'
    b_model_pth = r'E:\DX\MLPbaseline别动\siMLPe-原始\exps\baseline_h36m\log\bone_J_22\model-iter-45000.pth'

    model_position = Predictor(48, 50, 66, 22)
    model_bone = Predictor(48, 50, 66, 22)

    p_state_dict = torch.load(p_model_pth)
    model_position.load_state_dict(p_state_dict, strict=True)
    model_position.eval()
    model_position.cuda()

    b_state_dict = torch.load(b_model_pth)
    model_bone.load_state_dict(b_state_dict, strict=True)
    model_bone.eval()
    model_bone.cuda()

    return model_position, model_bone


def train_step(h36m_position_input, h36m_position_target, FusionModel, optimizer, nb_iter, total_iter, max_lr, min_lr) :

    orign = h36m_position_input.reshape(-1, 50, 25, 3)
    strat_idx = [0, 3, 4, 5, 0, 7, 8, 9, 0, 11, 12, 13, 12, 15, 16, 17, 17, 12, 20, 21, 22, 22]  # 22
    end_idx = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 22
    start_node = orign[:, :, strat_idx]
    end_node = orign[:, :, end_idx]
    bone_input = end_node - start_node
    bone_input = bone_input.reshape(bone_input.shape[0], bone_input.shape[1], -1)

    h36m_position_input = orign[:, :, 3:].reshape(h36m_position_input.shape[0], h36m_position_input.shape[1], -1)
    h36m_position_target = h36m_position_target.reshape(-1, 10, 25, 3)[:, :, 3:].reshape(h36m_position_target.shape[0],
                                                                                         h36m_position_target.shape[1], -1)

    bone_input_ = bone_input.clone()
    bone_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], bone_input_.cuda())

    h36m_position_input_ = h36m_position_input.clone()
    h36m_position_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_position_input_.cuda())

    bone_pred = model_bone(bone_input_.cuda())
    position_pred = model_position(h36m_position_input_.cuda())

    position_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], position_pred)
    bone_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], bone_pred)

    offset = h36m_position_input[:, -1:].cuda()
    position_pred = position_pred[:, :config.motion.h36m_target_length] + offset
    offset_ = bone_input[:, -1:].cuda()
    bone_pred = bone_pred[:, :config.motion.h36m_target_length] + offset_
    bone_pred = bone_pred.detach()
    position_pred = position_pred.detach()

    s = FusionModel(bone_pred, position_pred)
    motion_pred_fusion = fusion_C(bone_pred, position_pred, s)

    b,n,c = h36m_position_target.shape
    motion_pred_fusion = motion_pred_fusion.reshape(b,n,22,3).reshape(-1,3)
    h36m_position_target = h36m_position_target.cuda().reshape(b, n, 22, 3).reshape(-1, 3)
    rloss_fusion = torch.mean(torch.norm(motion_pred_fusion - h36m_position_target, 2, 1))

    motion_pred_fusion = motion_pred_fusion.reshape(b,n,22,3)
    dmotion_pred_fusion = gen_velocity(motion_pred_fusion)
    motion_gt = h36m_position_target.reshape(b, n, 22, 3)
    dmotion_gt = gen_velocity(motion_gt)
    dloss_fusion = torch.mean(torch.norm((dmotion_pred_fusion - dmotion_gt).reshape(-1,3), 2, 1))

    loss = rloss_fusion + dloss_fusion
    writer.add_scalar('Loss/iter', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


model_position, model_bone = get_bp_model()
FusionModel = FusionModel(10, 66)
FusionModel.train()
FusionModel.cuda()

config.motion.h36m_target_length = config.motion.h36m_target_length_train
dataset = H36MDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=0, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
eval_dataset = H36MEval(eval_config, 'test')

shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                        num_workers=0, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
optimizer = torch.optim.Adam(FusionModel.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < 2000:

    for (h36m_motion_input, h36m_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, FusionModel, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % 1 == 0:
            torch.save(FusionModel.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            FusionModel.eval()
            acc_tmp = test(eval_config, model_position, model_bone, FusionModel, eval_dataloader)
            print(acc_tmp)
            writer.add_scalar('MPJPE/iter', sum(acc_tmp), nb_iter)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            FusionModel.train()

        if (nb_iter + 1) == 2000:
            break
        nb_iter += 1

writer.close()

