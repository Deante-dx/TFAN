

def fusion_boneself(bone):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    output = bone.clone()
    output[:, :, 3] += output[:, :, 0]
    output[:, :, 6] += output[:, :, 1]
    output[:, :, 9] += output[:, :, 2]
    for i in range(4, 6):
        output[:, :, i] += output[:, :, i - 1]
    for i in range(7, 9):
        output[:, :, i] += output[:, :, i - 1]
    for i in range(10, 13):
        output[:, :, i] += output[:, :, i - 1]
    output[:, :, 13] += output[:, :, 9]
    for i in range(14, 18):
        output[:, :, i] += output[:, :, i - 1]
    output[:, :, 18] += output[:, :, 15]

    output[:, :, 19] += output[:, :, 9]
    for i in range(20, 24):
        output[:, :, i] += output[:, :, i - 1]
    output[:, :, 24] += output[:, :, 21]
    order = [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    output = output[:, :, order]

    return output


def fusion_A(bone, position):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    order = [0, 4, 8, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    position = position[:, :, order]
    output = bone.clone()
    output[:, :, 0] = (position[:, :, 0] + bone[:, :, 0]) / 2
    output[:, :, 1] = (position[:, :, 1] + bone[:, :, 1]) / 2
    output[:, :, 2] = (position[:, :, 2] + bone[:, :, 2]) / 2

    output[:, :, 3] += position[:, :, 0]
    output[:, :, 6] += position[:, :, 1]
    output[:, :, 9] += position[:, :, 2]
    for i in range(4, 6):
        output[:, :, i] += position[:, :, i - 1]
    for i in range(7, 9):
        output[:, :, i] += position[:, :, i - 1]
    for i in range(10, 13):
        output[:, :, i] += position[:, :, i - 1]
    output[:, :, 13] += position[:, :, 9]
    for i in range(14, 18):
        output[:, :, i] += position[:, :, i - 1]
    output[:, :, 18] += position[:, :, 15]
    output[:, :, 19] += position[:, :, 9]
    for i in range(20, 24):
        output[:, :, i] += position[:, :, i - 1]
    output[:, :, 24] += position[:, :, 21]
    order = [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    output = output[:, :, order]


    return output


def fusion_B(bone, position, s):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    order = [0, 4, 8, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    position = position[:, :, order]
    output = position.clone()
    output[:, :, 0] = (position[:, :, 0] + bone[:, :, 0]) / 2
    output[:, :, 1] = (position[:, :, 1] + bone[:, :, 1]) / 2
    output[:, :, 2] = (position[:, :, 2] + bone[:, :, 2]) / 2

    alpha = s[:, :, [0]].repeat(1, 1, 3)
    beta = 1 - alpha

    output[:, :, 3] = bone[:, :, 3] + alpha*position[:, :, 0] + beta*output[:, :, 0].clone()
    output[:, :, 6] = bone[:, :, 6] + alpha*position[:, :, 1] + beta*output[:, :, 1].clone()
    output[:, :, 9] = bone[:, :, 9] + alpha*position[:, :, 2] + beta*output[:, :, 2].clone()
    for i in range(4, 6):
        output[:, :, i] = bone[:, :, i] + alpha*position[:, :, i-1] + beta*output[:, :, i-1].clone()
    for i in range(7, 9):
        output[:, :, i] = bone[:, :, i] + alpha*position[:, :, i-1] + beta*output[:, :, i-1].clone()
    for i in range(10, 13):
        output[:, :, i] = bone[:, :, i] + alpha*position[:, :, i-1] + beta*output[:, :, i-1].clone()
    output[:, :, 13] = bone[:, :, 13] + alpha*position[:, :, 9] + beta*output[:, :, 9].clone()
    for i in range(14, 18):
        output[:, :, i] = bone[:, :, i] + alpha*position[:, :, i-1] + beta*output[:, :, i-1].clone()
    output[:, :, 18] = bone[:, :, 18] + alpha*position[:, :, 15] + beta*output[:, :, 15].clone()

    output[:, :, 19] = bone[:, :, 19] + alpha*position[:, :, 9] + beta*output[:, :, 9].clone()
    for i in range(20, 24):
        output[:, :, i] = bone[:, :, i] + alpha*position[:, :, i-1] + beta*output[:, :, i-1].clone()
    output[:, :, 24] = bone[:, :, 24] + alpha*position[:, :, 21] + beta*output[:, :, 21].clone()
    order = [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    output = output[:, :, order]

    return output


def fusion_C(bone, position, s):
    bone = bone.reshape(bone.shape[0], bone.shape[1], -1, 3)
    position = position.reshape(position.shape[0], position.shape[1], -1, 3)
    order = [0, 4, 8, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    position = position[:, :, order]
    output = position.clone()

    alpha = s[:, :, [0]].repeat(1, 1, 3)
    beta = s[:, :, [1]].repeat(1, 1, 3)

    output[:, :, 0] = (position[:, :, 0] + bone[:, :, 0]) / 2
    output[:, :, 1] = (position[:, :, 1] + bone[:, :, 1]) / 2
    output[:, :, 2] = (position[:, :, 2] + bone[:, :, 2]) / 2

    output[:, :, 3] = (bone[:, :, 3] + alpha * position[:, :, 0] + (1-alpha) * output[:, :, 0].clone())*(1-beta) + beta*position[:, :, 3]
    output[:, :, 6] = (bone[:, :, 6] + alpha * position[:, :, 1] + (1-alpha) * output[:, :, 1].clone())*(1-beta) + beta*position[:, :, 6]
    output[:, :, 9] = (bone[:, :, 9] + alpha * position[:, :, 2] + (1-alpha) * output[:, :, 2].clone())*(1-beta) + beta*position[:, :, 9]

    for i in range(4, 6):
        output[:, :, i] = (bone[:, :, i] + alpha * position[:, :, i-1] + (1-alpha) * output[:, :, i-1].clone())*(1-beta) + beta*position[:, :, i]
    for i in range(7, 9):
        output[:, :, i] = (bone[:, :, i] + alpha * position[:, :, i-1] + (1-alpha) * output[:, :, i-1].clone())*(1-beta) + beta*position[:, :, i]
    for i in range(10, 13):
        output[:, :, i] = (bone[:, :, i] + alpha * position[:, :, i-1] + (1-alpha) * output[:, :, i-1].clone())*(1-beta) + beta*position[:, :, i]
    output[:, :, 13] = (bone[:, :, 13] + alpha * position[:, :, 9] + (1-alpha) * output[:, :, 9].clone())*(1-beta) + beta*position[:, :, 13]
    for i in range(14, 18):
        output[:, :, i] = (bone[:, :, i] + alpha * position[:, :, i-1] + (1-alpha) * output[:, :, i-1].clone())*(1-beta) + beta*position[:, :, i]
    output[:, :, 18] = (bone[:, :, 18] + alpha * position[:, :, 15] + (1-alpha) * output[:, :, 15].clone())*(1-beta) + beta*position[:, :, 18]

    output[:, :, 19] = (bone[:, :, 19] + alpha * position[:, :, 9] + (1-alpha) * output[:, :, 9].clone())*(1-beta) + beta*position[:, :, 19]
    for i in range(20, 24):
        output[:, :, i] = (bone[:, :, i] + alpha * position[:, :, i-1] + (1-alpha) * output[:, :, i-1].clone())*(1-beta) + beta*position[:, :, i]
    output[:, :, 24] = (bone[:, :, 24] + alpha * position[:, :, 21] + (1-alpha) * output[:, :, 21].clone())*(1-beta) + beta*position[:, :, 24]
    order = [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    output = output[:, :, order]
    return output