import torch
import torch.nn as nn
from math import log, pi



class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.feats_hidden = args.feats_hidden
        self.feats_out = args.feats_out
        self.pred_len = args.pred_length

        self.pro1 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden * 2, self.feats_out)
        self.gru = nn.GRUCell(self.feats_hidden, self.feats_hidden)
        self.relu = nn.ReLU()


    def forward (self, dec_h):

        B, K, F = dec_h.shape
        forward_output = []
        forward_h = self.relu(self.pro1(dec_h)).view(-1, F)
        forward_input = self.relu(self.pro2(forward_h))

        for t in range(self.pred_len):
            forward_h = self.gru(forward_input, forward_h)
            forward_input = self.relu(self.pro2(forward_h))
            forward_traj = self.pro3(torch.cat([forward_input, dec_h.reshape(-1, F)], 1))
            forward_output.append(forward_traj.view(-1, K, forward_traj.shape[-1]))

        pred_traj = torch.stack(forward_output, dim=1)

        return pred_traj



class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.K = args.K

    def glow(self, z, log_det):

        log_p = -0.5 * log(2*pi) - 0.5 * (z ** 2)
        log_p_sum = torch.sum(log_p)
        loss = - (log_p_sum  + log_det)
        loss = loss/(z.size(0)*z.size(1)*z.size(2))

        return loss

    def traj(self, pred_traj, fur):

        fur = fur.unsqueeze(2).repeat(1, 1, self.K, 1)
        traj_rmse = torch.sqrt(torch.sum((pred_traj - fur) ** 2, dim=-1) + 1e-8).sum(dim=1)
        best_idx = torch.argmin(traj_rmse, dim=1)
        loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()

        return loss_traj
