import numpy as np
import copy
import torch

def Metrics(pred_traj, target_traj):
    K = pred_traj.shape[2]
    tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, K, 1))
    traj_ADE = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1).mean(1)
    traj_FDE = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1)[:, -1]
    ade = np.min(traj_ADE, axis=1).mean()
    fde = np.min(traj_FDE, axis=1).mean()

    return ade,fde

def get_node_index(seq_list):

    for idx, framenum in enumerate(seq_list):
        if idx == 0:
            node_indices = framenum > 0
        else:
            node_indices *= (framenum > 0)
    return node_indices

def update_batch_pednum(batch_pednum, ped_list):

    updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
    updated_batch_pednum = copy.deepcopy(batch_pednum)

    cumsum = np.cumsum(updated_batch_pednum_)
    new_ped = copy.deepcopy(ped_list).cpu().numpy()

    for idx, num in enumerate(cumsum):
        num = int(num)
        if idx == 0:
            updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
        else:
            updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

    return updated_batch_pednum

def mean_normalize_abs_input( node_abs, st_ed):

    node_abs = node_abs.permute(1, 0, 2)
    for st, ed in st_ed:
        mean_x = torch.mean(node_abs[st:ed, :, 0])
        mean_y = torch.mean(node_abs[st:ed, :, 1])

        node_abs[st:ed, :, 0] = ((node_abs[st:ed, :, 0] - mean_x))
        node_abs[st:ed, :, 1] = ((node_abs[st:ed, :, 1] - mean_y))

    return node_abs.permute(1, 0, 2)

def get_st_ed(batch_num):

    cumsum = torch.cumsum(batch_num, dim=0)
    st_ed = []
    for idx in range(1, cumsum.shape[0]):
        st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

    st_ed.insert(0, (0, int(cumsum[0])))

    return st_ed


def keep_full_tra(inputs):

    node_index = get_node_index(inputs[3])

    nodes_abs_ = inputs[0][:, node_index, :]
    nodes_norm = inputs[1][:, node_index, :]

    nei_lists = inputs[4][:, node_index, :]
    nei_list = nei_lists[:, :, node_index]
    batch_pednum = update_batch_pednum(inputs[6], node_index)
    st = get_st_ed(batch_pednum)
    nodes_abs = mean_normalize_abs_input(nodes_abs_, st)
    #nodes_abs = mean_normalize_abs_input_2(nodes_abs_)
    full_tra = (nodes_abs, nodes_norm, nei_list)

    return full_tra