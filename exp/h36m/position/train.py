import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
import copy
from config import config
from lib.datasets.h36m import H36MDataset
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import link_file, ensure_dir
from lib.datasets.h36m_eval import H36MEval
from models.Predictor import Predictor
from test import test
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm


def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter) :

    orign = h36m_motion_input.reshape(-1, 50, 25, 3)
    h36m_motion_input = orign[:, :, 3:].reshape(h36m_motion_input.shape[0], h36m_motion_input.shape[1], -1)
    h36m_motion_target = h36m_motion_target.reshape(-1, 10, 25, 3)[:, :, 3:].reshape(h36m_motion_target.shape[0],
                                                                                     h36m_motion_target.shape[1], -1)
    h36m_motion_input_ = h36m_motion_input.clone()

    h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.cuda())
    motion_pred = model(h36m_motion_input_.cuda())
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    offset = h36m_motion_input[:, -1:].cuda()
    motion_pred = motion_pred[:, :config.motion.h36m_target_length] + offset

    b, n, c = h36m_motion_target.shape
    motion_pred = motion_pred.reshape(b, n, 22, 3).reshape(-1, 3)
    h36m_motion_target = h36m_motion_target.cuda().reshape(b, n, 22, 3).reshape(-1, 3)
    rloss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1)) #MPJPE

    motion_pred = motion_pred.reshape(b, n, 22, 3)
    dmotion_pred = gen_velocity(motion_pred)
    motion_gt = h36m_motion_target.reshape(b, n, 22, 3)
    dmotion_gt = gen_velocity(motion_gt)
    dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1)) #velocity loss

    loss = rloss + dloss

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)
    return loss.item(), optimizer, current_lr

model = Predictor(48, 50, 66, 22)
model.train()
model.cuda()

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
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)


if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < config.cos_lr_total_iters:

    for (h36m_motion_input, h36m_motion_target) in dataloader:

        if nb_iter + 1 > 50000:
            model.change = True
        else:
            model.change = False
        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every == 0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % config.save_every ==  0 :
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            acc_tmp = test(eval_config, model, eval_dataloader)
            print(acc_tmp)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            model.train()

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
