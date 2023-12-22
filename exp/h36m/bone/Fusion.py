
def fusion_boneself(bone):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    output = bone.clone()

    for i in range(1, 4):
        output[:, :, i] = output[:, :, i] + output[:, :, i - 1]
    for i in range(5, 8):
        output[:, :, i] = output[:, :, i] + output[:, :, i - 1]
    for i in range(9, 12):
        output[:, :, i] = output[:, :, i] + output[:, :, i - 1]
    output[:, :, 12] = output[:, :, 12] + output[:, :, 9]
    for i in range(13, 16):
        output[:, :, i] = output[:, :, i] + output[:, :, i - 1]
    output[:, :, 16] = output[:, :, 16] + output[:, :, 14]
    output[:, :, 17] = output[:, :, 17] + output[:, :, 9]
    for i in range(18, 21):
        output[:, :, i] = output[:, :, i] + output[:, :, i - 1]
    output[:, :, 21] = output[:, :, 21] + output[:, :, 19]

    return output


def fusion_A(bone, position):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    output = position.clone()
    output[:, :, 0] = (position[:, :, 0] + bone[:, :, 0]) / 2
    output[:, :, 4] = (position[:, :, 4] + bone[:, :, 4]) / 2
    output[:, :, 8] = (position[:, :, 8] + bone[:, :, 8]) / 2

    for i in range(1, 4):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]
    for i in range(5, 8):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]
    for i in range(9, 12):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]
    output[:, :, 12] = bone[:, :, 12] + position[:, :, 9]
    for i in range(13, 16):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]
    output[:, :, 16] = bone[:, :, 16] + position[:, :, 14]
    output[:, :, 17] = bone[:, :, 17] + position[:, :, 9]
    for i in range(18, 21):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]
    output[:, :, 21] = bone[:, :, 21] + position[:, :, 19]

    return output


def fusion_B(bone, position, s):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    output = position.clone()
    output[:, :, 0] = (position[:, :, 0] + bone[:, :, 0]) / 2
    output[:, :, 4] = (position[:, :, 4] + bone[:, :, 4]) / 2
    output[:, :, 8] = (position[:, :, 8] + bone[:, :, 8]) / 2

    alpha = s[:, :, [0]].repeat(1, 1, 3)
    beta = 1 - alpha
    for i in range(1, 4):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]*alpha + output[:, :, i - 1].clone()*beta
    for i in range(5, 8):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]*alpha + output[:, :, i - 1].clone()*beta
    for i in range(9, 12):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]*alpha + output[:, :, i - 1].clone()*beta
    output[:, :, 12] = bone[:, :, 12] + position[:, :, 9]*alpha + output[:, :, 9].clone()*beta
    for i in range(13, 16):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]*alpha + output[:, :, i - 1].clone()*beta
    output[:, :, 16] = bone[:, :, 16] + position[:, :, 14]*alpha + output[:, :, 14].clone()*beta
    output[:, :, 17] = bone[:, :, 17] + position[:, :, 9]*alpha + output[:, :, 9].clone()*beta
    for i in range(18, 21):
        output[:, :, i] = bone[:, :, i] + position[:, :, i - 1]*alpha + output[:, :, i - 1].clone()*beta
    output[:, :, 21] = bone[:, :, 21] + position[:, :, 19]*alpha + output[:, :, 19].clone()*beta

    return output


def fusion_C(bone, position, s):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    output = position.clone()
    alpha = s[:, :, [0]].repeat(1, 1, 3)
    beta = s[:, :, [1]].repeat(1, 1, 3)

    a = 0.5
    b = 1 - a
    output[:, :, 0] = a*position[:, :, 0] + b*bone[:, :, 0]
    output[:, :, 4] = a*position[:, :, 4] + b*bone[:, :, 4]
    output[:, :, 8] = a*position[:, :, 8] + b*bone[:, :, 8]

    for i in range(1, 4):
        output[:, :, i] = (bone[:, :, i] + (position[:, :, i - 1] * alpha + output[:, :, i - 1].clone() * (1-alpha))) * (1 - beta) + position[:, :, i] * beta
    for i in range(5, 8):
        output[:, :, i] = (bone[:, :, i] + (position[:, :, i - 1] * alpha + output[:, :, i - 1].clone() * (1-alpha))) * (1 - beta) + position[:, :, i] * beta
    for i in range(9, 12):
        output[:, :, i] = (bone[:, :, i] + (position[:, :, i - 1] * alpha + output[:, :, i - 1].clone() * (1-alpha))) * (1 - beta) + position[:, :, i] * beta
    output[:, :, 12] = (bone[:, :, 12] + (position[:, :, 9]*alpha + output[:, :, 9].clone()*(1-alpha))) * (1 - beta) + position[:, :, 12] * beta
    for i in range(13, 16):
        output[:, :, i] = (bone[:, :, i] + (position[:, :, i - 1] * alpha + output[:, :, i - 1].clone() * (1-alpha))) * (1 - beta) + position[:, :, i] * beta
    output[:, :, 16] = (bone[:, :, 16] + (position[:, :, 14]*alpha + output[:, :, 14].clone()*(1-alpha))) * (1 - beta) + position[:, :, 16] * beta
    output[:, :, 17] = (bone[:, :, 17] + (position[:, :, 9]*alpha + output[:, :, 9].clone()*(1-alpha))) * (1 - beta) + position[:, :, 17] * beta
    for i in range(18, 21):
        output[:, :, i] = (bone[:, :, i] + (position[:, :, i - 1] * alpha + output[:, :, i - 1].clone() * (1-alpha))) * (1 - beta) + position[:, :, i] * beta
    output[:, :, 21] = (bone[:, :, 21] + (position[:, :, 19]*alpha + output[:, :, 19].clone()*(1-alpha))) * (1 - beta) + position[:, :, 21] * beta

    return output

