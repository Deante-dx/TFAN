

def fusion_boneself(bone):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    output = bone.clone()
    output[:, :, 3] = output[:, :, 3] + output[:, :, 0]
    output[:, :, 4] = output[:, :, 4] + output[:, :, 1]
    output[:, :, 5] = output[:, :, 5] + output[:, :, 2]
    output[:, :, 6] = output[:, :, 6] + output[:, :, 3]
    output[:, :, 7] = output[:, :, 7] + output[:, :, 4]
    output[:, :, 8] = output[:, :, 8] + output[:, :, 5]
    output[:, :, 9] = output[:, :, 9] + output[:, :, 5]
    output[:, :, 10] = output[:, :, 10] + output[:, :, 5]
    output[:, :, 11] = output[:, :, 11] + output[:, :, 8]
    output[:, :, 12] = output[:, :, 12] + output[:, :, 9]
    output[:, :, 13] = output[:, :, 13] + output[:, :, 10]
    output[:, :, 14] = output[:, :, 14] + output[:, :, 12]
    output[:, :, 15] = output[:, :, 15] + output[:, :, 13]
    output[:, :, 16] = output[:, :, 16] + output[:, :, 14]
    output[:, :, 17] = output[:, :, 17] + output[:, :, 15]

    return output


def fusion_A(bone, position):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    output = position.clone()
    output[:, :, 0] = (position[:, :, 0] + bone[:, :, 0]) / 2
    output[:, :, 1] = (position[:, :, 1] + bone[:, :, 1]) / 2
    output[:, :, 2] = (position[:, :, 2] + bone[:, :, 2]) / 2
    output[:, :, 3] = bone[:, :, 3] + position[:, :, 0]
    output[:, :, 4] = bone[:, :, 4] + position[:, :, 1]
    output[:, :, 5] = bone[:, :, 5] + position[:, :, 2]
    output[:, :, 6] = bone[:, :, 6] + position[:, :, 3]
    output[:, :, 7] = bone[:, :, 7] + position[:, :, 4]
    output[:, :, 8] = bone[:, :, 8] + position[:, :, 5]
    output[:, :, 9] = bone[:, :, 9] + position[:, :, 5]
    output[:, :, 10] = bone[:, :, 10] + position[:, :, 5]
    output[:, :, 11] = bone[:, :, 11] + position[:, :, 8]
    output[:, :, 12] = bone[:, :, 12] + position[:, :, 9]
    output[:, :, 13] = bone[:, :, 13] + position[:, :, 10]
    output[:, :, 14] = bone[:, :, 14] + position[:, :, 12]
    output[:, :, 15] = bone[:, :, 15] + position[:, :, 13]
    output[:, :, 16] = bone[:, :, 16] + position[:, :, 14]
    output[:, :, 17] = bone[:, :, 17] + position[:, :, 15]

    return output


def fusion_B(bone, position, s):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    output = position.clone()
    output[:, :, 0] = (position[:, :, 0] + bone[:, :, 0]) / 2
    output[:, :, 1] = (position[:, :, 1] + bone[:, :, 1]) / 2
    output[:, :, 2] = (position[:, :, 2] + bone[:, :, 2]) / 2

    alpha = s[:, :, [0]].repeat(1, 1, 3)
    beta = 1 - alpha

    output[:, :, 3] = bone[:, :, 3] + position[:, :, 0] * alpha + output[:, :, 0].clone() * beta
    output[:, :, 4] = bone[:, :, 4] + position[:, :, 1] * alpha + output[:, :, 1].clone() * beta
    output[:, :, 5] = bone[:, :, 5] + position[:, :, 2] * alpha + output[:, :, 2].clone() * beta
    output[:, :, 6] = bone[:, :, 6] + position[:, :, 3] * alpha + output[:, :, 3].clone() * beta
    output[:, :, 7] = bone[:, :, 7] + position[:, :, 4] * alpha + output[:, :, 4].clone() * beta
    output[:, :, 8] = bone[:, :, 8] + position[:, :, 5] * alpha + output[:, :, 5].clone() * beta
    output[:, :, 9] = bone[:, :, 9] + position[:, :, 5] * alpha + output[:, :, 5].clone() * beta
    output[:, :, 10] = bone[:, :, 10] + position[:, :, 5] * alpha + output[:, :, 5].clone() * beta
    output[:, :, 11] = bone[:, :, 11] + position[:, :, 8] * alpha + output[:, :, 8].clone() * beta
    output[:, :, 12] = bone[:, :, 12] + position[:, :, 9] * alpha + output[:, :, 9].clone() * beta
    output[:, :, 13] = bone[:, :, 13] + position[:, :, 10] * alpha + output[:, :, 10].clone() * beta
    output[:, :, 14] = bone[:, :, 14] + position[:, :, 12] * alpha + output[:, :, 12].clone() * beta
    output[:, :, 15] = bone[:, :, 15] + position[:, :, 13] * alpha + output[:, :, 13].clone() * beta
    output[:, :, 16] = bone[:, :, 16] + position[:, :, 14] * alpha + output[:, :, 14].clone() * beta
    output[:, :, 17] = bone[:, :, 17] + position[:, :, 15] * alpha + output[:, :, 15].clone() * beta

    return output


def fusion_C(bone, position, s):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    output = position.clone()
    alpha = s[:, :, [0]].repeat(1, 1, 3)
    beta = s[:, :, [1]].repeat(1, 1, 3)

    output[:, :, 0] = (position[:, :, 0] + bone[:, :, 0]) / 2
    output[:, :, 1] = (position[:, :, 1] + bone[:, :, 1]) / 2
    output[:, :, 2] = (position[:, :, 2] + bone[:, :, 2]) / 2

    output[:, :, 3] = (bone[:, :, 3] + position[:, :, 0] * alpha + output[:, :, 0].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 3] * beta
    output[:, :, 4] = (bone[:, :, 4] + position[:, :, 1] * alpha + output[:, :, 1].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 4] * beta
    output[:, :, 5] = (bone[:, :, 5] + position[:, :, 2] * alpha + output[:, :, 2].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 5] * beta
    output[:, :, 6] = (bone[:, :, 6] + position[:, :, 3] * alpha + output[:, :, 3].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 6] * beta
    output[:, :, 7] = (bone[:, :, 7] + position[:, :, 4] * alpha + output[:, :, 4].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 7] * beta
    output[:, :, 8] = (bone[:, :, 8] + position[:, :, 5] * alpha + output[:, :, 5].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 8] * beta
    output[:, :, 9] = (bone[:, :, 9] + position[:, :, 5] * alpha + output[:, :, 5].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 9] * beta
    output[:, :, 10] = (bone[:, :, 10] + position[:, :, 5] * alpha + output[:, :, 5].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 10] * beta
    output[:, :, 11] = (bone[:, :, 11] + position[:, :, 8] * alpha + output[:, :, 8].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 11] * beta
    output[:, :, 12] = (bone[:, :, 12] + position[:, :, 9] * alpha + output[:, :, 9].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 12] * beta
    output[:, :, 13] = (bone[:, :, 13] + position[:, :, 10] * alpha + output[:, :, 10].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 13] * beta
    output[:, :, 14] = (bone[:, :, 14] + position[:, :, 12] * alpha + output[:, :, 12].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 14] * beta
    output[:, :, 15] = (bone[:, :, 15] + position[:, :, 13] * alpha + output[:, :, 13].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 15] * beta
    output[:, :, 16] = (bone[:, :, 16] + position[:, :, 14] * alpha + output[:, :, 14].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 16] * beta
    output[:, :, 17] = (bone[:, :, 17] + position[:, :, 15] * alpha + output[:, :, 15].clone() * (1 - alpha)) * (1 - beta) + position[:, :, 17] * beta

    return output