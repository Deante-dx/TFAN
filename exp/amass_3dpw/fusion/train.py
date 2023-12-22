import os
import json
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from config import config
from lib.datasets.amass import AMASSDataset
from lib.datasets.amass_eval import AMASSEval
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import link_file, ensure_dir
from models.Predictor import Predictor
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops.layers.torch import Rearrange
from models.FusionModel import FusionModel
from Fusion import *
from test import test
from lib.datasets.pw3d_eval import PW3DEval


torch.use_deterministic_algorithms(True)
acc_log = open('fusion_log', 'a')
torch.manual_seed(888)
writer = SummaryWriter()
acc_log.write(''.join('Seed : ' + str(888) + '\n'))

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

dct_m,idct_m = get_dct_matrix(config.motion.amass_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
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
    p_model_pth = r'E:\DX\MLPbaseline别动\siMLPe-原始\exps\baseline_amass\log\position_J_54\model-iter-115000.pth'
    b_model_pth = r'E:\DX\MLPbaseline别动\siMLPe-原始\exps\baseline_amass\log\bone_J_54\model-iter-115000.pth'

    model_position = Predictor(48, 50, 54, 54)
    model_bone = Predictor(48, 50, 54, 54)

    p_state_dict = torch.load(p_model_pth)
    model_position.load_state_dict(p_state_dict, strict=True)
    model_position.eval()
    model_position.cuda()

    b_state_dict = torch.load(b_model_pth)
    model_bone.load_state_dict(b_state_dict, strict=True)
    model_bone.eval()
    model_bone.cuda()

    return model_position, model_bone


def train_step(amass_motion_input, amass_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :
    in_origin = amass_motion_input.reshape(-1, 50, 18, 3)
    start_idx = [0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10, 12, 13, 14, 15]
    end_idx = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    bone = in_origin[:, :, end_idx] - in_origin[:, :, start_idx]
    bone_input = torch.cat([in_origin[:, :, [0, 1, 2]], bone], dim=2)
    bone_input = bone_input.reshape(bone_input.shape[0], bone_input.shape[1], -1)
    out_origin = amass_motion_target.reshape(-1, 25, 18, 3)
    bone = out_origin[:, :, end_idx] - out_origin[:, :, start_idx]
    bone_target = torch.cat([out_origin[:, :, [0, 1, 2]], bone], dim=2)
    # bone_output = bone_output.reshape(bone_output.shape[0], bone_output.shape[1], -1)

    if config.deriv_input:
        b,n,c = amass_motion_input.shape
        bone_input_ = bone_input.clone()
        bone_input_ = torch.matmul(dct_m, bone_input_.cuda())

        amass_motion_input_ = amass_motion_input.clone()
        amass_motion_input_ = torch.matmul(dct_m, amass_motion_input_.cuda())
    else:
        amass_motion_input_ = amass_motion_input.clone()
        bone_input_ = bone_input.clone()

    bone_pred = model_bone(bone_input_.cuda())
    position_pred = model_position(amass_motion_input_.cuda())
    position_pred = torch.matmul(idct_m, position_pred)
    bone_pred = torch.matmul(idct_m, bone_pred)

    if config.deriv_output:
        offset = amass_motion_input[:, -1:].cuda()
        position_pred = position_pred[:, :25] + offset
        offset_ = bone_input[:, -1:].cuda()
        bone_pred = bone_pred[:, :25] + offset_
        bone_pred = bone_pred.detach()
        position_pred = position_pred.detach()

    s = model(bone_pred, position_pred)
    motion_pred_fusion = fusion_C(bone_pred, position_pred, s)

    b, n, c = amass_motion_target.shape
    motion_pred_fusion = motion_pred_fusion.reshape(b, n, 18, 3).reshape(-1, 3)
    amass_motion_target = amass_motion_target.cuda().reshape(b, n, 18, 3).reshape(-1, 3)
    loss_fusion = torch.mean(torch.norm(motion_pred_fusion - amass_motion_target, 2, 1))

    motion_pred_fusion = motion_pred_fusion.reshape(b, n, 18, 3)
    dmotion_pred_fusion = gen_velocity(motion_pred_fusion)
    motion_gt = amass_motion_target.reshape(b, n, 18, 3)
    dmotion_gt = gen_velocity(motion_gt)
    dloss_fusion = torch.mean(torch.norm((dmotion_pred_fusion - dmotion_gt).reshape(-1, 3), 2, 1))

    loss = loss_fusion + dloss_fusion
    writer.add_scalar('Loss/iter', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


model_position, model_bone = get_bp_model()

model = FusionModel(25, 54)
model.train()
model.cuda()

config.motion.amass_target_length = config.motion.amass_target_length_train
dataset = AMASSDataset(config, 'train', config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=0, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_dataset = AMASSEval(config, 'test')
shuffle = False
sampler = None
dataloader_eval = DataLoader(eval_dataset, batch_size=128,
                        num_workers=0, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)



ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < 2000:

    for (amass_motion_input, amass_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(amass_motion_input, amass_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
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

        if (nb_iter + 1) == 2000:
            break
        nb_iter += 1

writer.close()