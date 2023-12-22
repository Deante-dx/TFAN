import os
import json
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from config import config
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import link_file, ensure_dir
from models.Predictor import Predictor
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.FusionModel import FusionModel
from Fusion import *
from test import test
from lib.datasets.cmu import CMU_Motion3D

torch.use_deterministic_algorithms(True)
acc_log = open(config.test, 'a')
acc_log.write(''.join('Seed : ' + str(config.seed) + '\n'))
torch.manual_seed(config.seed)
writer = SummaryWriter()

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

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

dct_m,idct_m = get_dct_matrix(config.motion.cmu_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, optimizer) :
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


def get_bp_model():
    p_model_pth = r'E:\DX\MLPbaseline别动\siMLPe-原始\exps\baseline_cmu\log\J_25\model-iter-45000.pth'
    b_model_pth = r'E:\DX\MLPbaseline别动\siMLPe-原始\exps\baseline_cmu\log\bone_J_25\bone_model-iter-43000.pth'

    model_position = Predictor(48, 50, 75, 25)
    model_bone = Predictor(48, 50, 75, 25)

    p_state_dict = torch.load(p_model_pth)
    model_position.load_state_dict(p_state_dict, strict=True)
    model_position.eval()
    model_position.cuda()

    b_state_dict = torch.load(b_model_pth)
    model_bone.load_state_dict(b_state_dict, strict=True)
    model_bone.eval()
    model_bone.cuda()

    return model_position, model_bone


def train_step(cmu_motion_input, cmu_motion_target, model, optimizer, nb_iter):
    in_origin = cmu_motion_input.reshape(-1, 50, 25, 3)
    start_idx = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 9, 13, 14, 15, 16, 15, 9, 19, 20, 21, 22, 21]
    end_idx = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    bone = in_origin[:, :, end_idx] - in_origin[:, :, start_idx]
    bone_input = torch.cat([in_origin[:, :, [0, 4, 8]], bone], dim=2)
    bone_input = bone_input.reshape(bone_input.shape[0], bone_input.shape[1], -1)
    out_origin = cmu_motion_target.reshape(-1, 10, 25, 3)
    bone = out_origin[:, :, end_idx] - out_origin[:, :, start_idx]
    bone_target = torch.cat([out_origin[:, :, [0, 4, 8]], bone], dim=2)

    bone_input_ = bone_input.clone()
    bone_input_ = torch.matmul(dct_m, bone_input_.cuda())
    cmu_motion_input_ = cmu_motion_input.clone()
    cmu_motion_input_ = torch.matmul(dct_m, cmu_motion_input_.cuda())

    bone_pred = model_bone(bone_input_.cuda())
    position_pred = model_position(cmu_motion_input_.cuda())
    position_pred = torch.matmul(idct_m, position_pred)
    bone_pred = torch.matmul(idct_m, bone_pred)

    offset = cmu_motion_input[:, -1:].cuda()
    position_pred = position_pred[:, :10] + offset
    offset_ = bone_input[:, -1:].cuda()
    bone_pred = bone_pred[:, :10] + offset_
    bone_pred = bone_pred.detach()
    position_pred = position_pred.detach()

    s = model(bone_pred, position_pred)
    motion_pred_fusion = fusion_C(bone_pred, position_pred, s)

    b, n, c = cmu_motion_target.shape
    motion_pred_fusion = motion_pred_fusion.reshape(b, n, 25, 3).reshape(-1, 3)
    cmu_motion_target = cmu_motion_target.cuda().reshape(b, n, 25, 3).reshape(-1, 3)
    loss_fusion = torch.mean(torch.norm(motion_pred_fusion - cmu_motion_target, 2, 1))

    motion_pred_fusion = motion_pred_fusion.reshape(b, n, 25, 3)
    dmotion_pred_fusion = gen_velocity(motion_pred_fusion)
    motion_gt = cmu_motion_target.reshape(b, n, 25, 3)
    dmotion_gt = gen_velocity(motion_gt)
    dloss_fusion = torch.mean(torch.norm((dmotion_pred_fusion - dmotion_gt).reshape(-1, 3), 2, 1))

    loss = loss_fusion + dloss_fusion
    writer.add_scalar('Loss/iter', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


model_position, model_bone = get_bp_model()

model = FusionModel(25, 75)
model.train()
model.cuda()

config.motion.cmu_target_length = config.motion.cmu_target_length_train
dataset = CMU_Motion3D(path_to_data=config.h36m_anno_dir, actions='all', input_n=50, output_n=10, split=0)


shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=0, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_dataset = CMU_Motion3D(path_to_data=config.h36m_anno_dir, actions='all', input_n=50, output_n=25, split=1)

shuffle = False
sampler = None
dataloader_eval = DataLoader(eval_dataset, batch_size=128,
                        num_workers=0, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < 2000:

    for (cmu_motion_input, cmu_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(cmu_motion_input, cmu_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every == 0:
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % 500 == 0:
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            acc_tmp = test(model, model_position, model_bone, dataloader_eval)
            print(acc_tmp)
            writer.add_scalar('MPJPE/iter', sum(acc_tmp), nb_iter)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            model.train()

        if (nb_iter + 1) == 2000 :
            break
        nb_iter += 1

writer.close()