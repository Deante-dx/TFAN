from lib.datasets.cmu import CMU_Motion3D
import numpy as np
from config  import config
from models.Predictor import Predictor
from Fusion import *
import torch
from torch.utils.data import DataLoader
import os.path as osp

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

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

dct_m,idct_m = get_dct_matrix(config.motion.cmu_input_length)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def regress_pred(model, pbar, num_samples, m_p3d_h36):
    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c = motion_input.shape
        num_samples += b
        motion_input = motion_input.reshape(b, n, 25, 3)

        in_origin = motion_input.reshape(-1, 50, 25, 3)
        start_idx = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 9, 13, 14, 15, 16, 15, 9, 19, 20, 21, 22, 21]
        end_idx = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        bone = in_origin[:, :, end_idx] - in_origin[:, :, start_idx]
        bone_input = torch.cat([in_origin[:, :, [0, 4, 8]], bone], dim=2)
        bone_input = bone_input.reshape(bone_input.shape[0], bone_input.shape[1], -1)

        motion_input = motion_input.reshape(b, n, -1)
        outputs = []
        step = config.motion.cmu_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                bone_input_ = bone_input.clone()
                bone_input_ = torch.matmul(dct_m, bone_input_.cuda())

                bone_pred = model(bone_input_)
                bone_pred = torch.matmul(idct_m, bone_pred)[:, :step, :]
                bone_pred = bone_pred + bone_input[:, -1:, :].repeat(1, step, 1)
                output = bone_pred.clone()
                output = fusion_boneself(output)

            output = output.reshape(-1, 25*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            bone_input = torch.cat([bone_input[:, step:].cuda(), bone_pred], dim=1)
        motion_pred = torch.cat(outputs, dim=1)[:,:25]

        b,n,c = motion_target.shape
        motion_target = motion_target.detach().reshape(b, n, 25, 3)
        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        motion_pred = motion_pred.reshape(b, n, 25, 3)

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(model, dataloader) :

    m_p3d_h36 = np.zeros([25])
    titles = np.array(range(25)) + 1
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, m_p3d_h36)

    ret = {}
    for j in range(25):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]


if __name__ == "__main__":
    pth = osp.join(config.root_dir, 'checkpoints/cmu/model_bone.pth')
    model = Predictor(48, 50, 75, 25)
    state_dict = torch.load(pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.cmu_target_length = config.motion.cmu_target_length_eval
    dataset = CMU_Motion3D(path_to_data=config.h36m_anno_dir, actions='all', input_n=50, output_n=25, split=1)


    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=0, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    print(test(model, dataloader))

