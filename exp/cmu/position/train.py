import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
from config import config
from lib.datasets.cmu import CMU_Motion3D
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import link_file, ensure_dir
from test import test
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.Predictor import Predictor

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


dct_m, idct_m = get_dct_matrix(config.motion.cmu_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def update_lr_multistep(nb_iter, optimizer):
    if nb_iter > 40000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm


def train_step(cmu_motion_input, cmu_motion_target, model, optimizer, nb_iter):

    cmu_motion_input_ = cmu_motion_input.clone()
    cmu_motion_input_ = torch.matmul(dct_m, cmu_motion_input_.cuda())

    motion_pred = model(cmu_motion_input_.cuda())
    motion_pred = torch.matmul(idct_m, motion_pred)

    offset = cmu_motion_input[:, -1:].cuda()
    motion_pred = motion_pred[:, :config.motion.cmu_target_length] + offset

    b, n, c = cmu_motion_target.shape
    motion_pred = motion_pred.reshape(b, n, 25, 3).reshape(-1, 3)
    cmu_motion_target = cmu_motion_target.cuda().reshape(b, n, 25, 3).reshape(-1, 3)
    rloss = torch.mean(torch.norm(motion_pred - cmu_motion_target, 2, 1))


    motion_pred = motion_pred.reshape(b, n, 25, 3)
    dmotion_pred = gen_velocity(motion_pred)
    motion_gt = cmu_motion_target.reshape(b, n, 25, 3)
    dmotion_gt = gen_velocity(motion_gt)
    dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
    loss = rloss + dloss

    writer.add_scalar('Loss/iter', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, optimizer)
    writer.add_scalar('LR/iter', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


model = Predictor(48, 50, 75, 25)
model.train()
model.cuda()

config.motion.cmu_target_length = config.motion.cmu_target_length_train
dataset = CMU_Motion3D(path_to_data=config.h36m_anno_dir, actions='all', input_n=50,
                       output_n=10, split=0)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=0, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_dataset = CMU_Motion3D(path_to_data=config.h36m_anno_dir, actions='all', input_n=50,
                            output_n=25, split=1)

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

while (nb_iter + 1) < config.cos_lr_total_iters:

    for (cmu_motion_input, cmu_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(cmu_motion_input, cmu_motion_target, model, optimizer, nb_iter)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every == 0:
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % config.save_every == 0:
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            acc_tmp = test(model, dataloader_eval)
            print(acc_tmp)
            writer.add_scalar('MPJPE/iter', sum(acc_tmp), nb_iter)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            model.train()

        if (nb_iter + 1) == config.cos_lr_total_iters:
            break
        nb_iter += 1

writer.close()
