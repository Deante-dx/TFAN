import numpy as np
from config import config
from models.Predictor import Predictor
from lib.datasets.h36m_eval import H36MEval
import torch
from torch.utils.data import DataLoader
from Fusion import *
from models.FusionModel import FusionModel
import os.path as osp


results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

def get_bp_model():
    p_model_pth = osp.join(config.root_dir, 'checkpoints/h36m/model_position.pth')
    b_model_pth = osp.join(config.root_dir, 'checkpoints/h36m/model_bone.pth')

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




def regress_pred(model_position, model_bone, Model, pbar, num_samples, joint_used_xyz, m_p3d_h36):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)
    joint_bone = np.array([0, 1, 6, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]).astype(np.int64)

    for (motion_input, motion_target) in pbar:
        bone_ori = motion_input.reshape(motion_input.shape[0], motion_input.shape[1], 32, 3)[:, :, joint_bone]
        bone = bone_ori.clone()
        strat_idx = [0, 3, 4, 5, 0, 7, 8, 9, 0, 11, 12, 13, 12, 15, 16, 17, 17, 12, 20, 21, 22, 22]  # 22
        end_idx = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 22
        start_node = bone[:, :, strat_idx]
        end_node = bone[:, :, end_idx]
        bone_input = end_node - start_node
        bone_input = bone_input.reshape(bone_input.shape[0], bone_input.shape[1], -1)
        motion_input = motion_input.cuda()
        b,n,c,_ = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        input = motion_input.clone().cpu()   #用于可视化
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                motion_input_ = motion_input.clone()
                motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
                bone_input_ = bone_input.clone()
                bone_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], bone_input_.cuda())

                position_pred = model_position(motion_input_.cuda())
                bone_pred = model_bone(bone_input_.cuda())

                position_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], position_pred)[:, :step, :]
                bone_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], bone_pred)[:, :step]

                offset_ = bone_input[:, -1:].cuda()
                bone_pred = bone_pred[:, :config.motion.h36m_target_length] + offset_
                position_pred = position_pred + motion_input[:, -1:, :].repeat(1, step, 1)

                s = Model(bone_pred.detach(), position_pred.detach())
                output = fusion_C(bone_pred, position_pred, s) #9.2, 20.8, 44.5, 55.3, 73.9, 88.3, 100.1, 108.0

                output = output.reshape(output.shape[0], output.shape[1], 22, 3)
                output_ = torch.cat([bone_ori[:, :step, :3].cuda(), output], dim=2)
                bone_out = output_[:, :, end_idx] - output_[:, :, strat_idx]
            bone_out = bone_out.reshape(b, step, -1)
            output = output.reshape(b, step, -1)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], dim=1)
            bone_input = torch.cat([bone_input[:, step:].cuda(), bone_out], dim=1)

        motion_pred = torch.cat(outputs, dim=1)[:, :25]
        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape
        motion_gt = motion_target.clone()
        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]
        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36


def test(config, model_position, model_bone, Model, dataloader):
    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(model_position, model_bone, Model, pbar, num_samples, joint_used_xyz, m_p3d_h36)

    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]


if __name__ == "__main__":
    model_position, model_bone = get_bp_model()
    Model = FusionModel(10, 66).cuda()
    fusion_state_dict = torch.load(osp.join(config.root_dir, 'checkpoints/h36m/model_fusion.pth'))
    Model.load_state_dict(fusion_state_dict)
    Model.eval()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    action = ["walking", "eating", "smoking", "discussion", "directions",
     "greeting", "phoning", "posing", "purchases", "sitting",
     "sittingdown", "takingphoto", "waiting", "walkingdog",
     "walkingtogether", "all"]
    act_idx = -1
    dataset = H36MEval(config, 'test', act=action[act_idx])

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)
    print(test(config, model_position, model_bone, Model, dataloader))
