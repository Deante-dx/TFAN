from models.Predictor import Predictor
import numpy as np
from tqdm import tqdm
from config  import config
from lib.datasets.amass_eval import AMASSEval
import torch
from torch.utils.data import DataLoader
from Fusion import *
from models.FusionModel import FusionModel
from lib.datasets.pw3d_eval import PW3DEval
import os.path as osp

def get_bp_model():
    p_model_pth = osp.join(config.root_dir, 'checkpoints/amass_3dpw/model_position.pth')
    b_model_pth = osp.join(config.root_dir, 'checkpoints/amass_3dpw/model_bone.pth')
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

dct_m,idct_m = get_dct_matrix(config.motion.amass_input_length)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def regress_pred(model, model_position, model_bone, pbar, num_samples, m_p3d_h36):

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 18, 3)

        in_origin = motion_input.reshape(-1, 50, 18, 3)
        start_idx = [0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10, 12, 13, 14, 15]
        end_idx = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        bone = in_origin[:, :, end_idx] - in_origin[:, :, start_idx]
        bone_input = torch.cat([in_origin[:, :, [0, 1, 2]], bone], dim=2)
        bone_input = bone_input.reshape(bone_input.shape[0], bone_input.shape[1], -1)

        motion_input = motion_input.reshape(b, n, -1)
        outputs = []
        step = config.motion.amass_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():

                bone_input_ = bone_input.clone()
                bone_input_ = torch.matmul(dct_m, bone_input_.cuda())
                bone_input_ = bone_input_[:, -config.motion.amass_input_length:]
                motion_input_ = motion_input.clone()
                motion_input_ = torch.matmul(dct_m, motion_input_.cuda())
                motion_input_ = motion_input_[:, -config.motion.amass_input_length:]

                position_pred = model_position(motion_input_.cuda())
                bone_pred = model_bone(bone_input_.cuda())

                position_pred = torch.matmul(idct_m, position_pred)[:, :step, :]
                bone_pred = torch.matmul(idct_m, bone_pred)[:, :step]

                offset_ = bone_input[:, -1:].cuda()
                bone_pred = bone_pred[:, :config.motion.amass_target_length] + offset_
                position_pred = position_pred + motion_input[:, -1:, :].repeat(1, step, 1)

                s = model(bone_pred.detach(), position_pred.detach())
                output = fusion_C(bone_pred, position_pred, s)  # 10.6, 19.1, 33.9, 40.1, 50.0, 57.0, 62.2, 66.0 -> [10.5, 19.0, 33.7, 39.8, 49.7, 56.8, 62.0, 65.9]

                output = output.reshape(output.shape[0], output.shape[1], 18, 3)
                bone_out = output[:, :, end_idx] - output[:, :, start_idx]
                bone_out = torch.cat([output[:, :, [0, 1, 2]], bone_out], dim=2)
            bone_out = bone_out.reshape(b, step, -1)
            output = output.reshape(b, step, -1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], dim=1)
            bone_input = torch.cat([bone_input[:, step:].cuda(), bone_out], dim=1)
        motion_pred = torch.cat(outputs, dim=1)[:, :25]
        b,n,c = motion_target.shape
        motion_target = motion_target.detach().reshape(b, n, 18, 3)
        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        motion_pred = motion_pred.reshape(b, n, 18, 3)

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(model, model_position, model_bone, dataloader) :

    m_p3d_h36 = np.zeros([config.motion.amass_target_length])
    titles = np.array(range(config.motion.amass_target_length)) + 1
    num_samples = 0

    pbar = tqdm(dataloader)
    m_p3d_h36 = regress_pred(model, model_position, model_bone, pbar, num_samples, m_p3d_h36)

    ret = {}
    for j in range(config.motion.amass_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return ([round(ret[key][0], 1) for key in results_keys])


if __name__ == "__main__":
    model_position, model_bone = get_bp_model()
    model = FusionModel(10, 54)
    state_dict = torch.load(osp.join(config.root_dir, 'checkpoints/amass_3dpw/model_fusion.pth'))
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    config.motion.amass_target_length = config.motion.amass_target_length_eval


    dataset = AMASSEval(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=0, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    # config.motion.pw3d_target_length = config.motion.pw3d_target_length_eval
    # dataset = PW3DEval(config, 'test')
    #
    # shuffle = False
    # sampler = None
    # train_sampler = None
    # dataloader = DataLoader(dataset, batch_size=128,
    #                              num_workers=0, drop_last=False,
    #                              sampler=sampler, shuffle=shuffle, pin_memory=True)

    print(test(model, model_position, model_bone, dataloader))

