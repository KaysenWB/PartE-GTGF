import torch.nn as nn
from .ts_transformer import TS_transformer
from .decoder_loss import Decoder, Loss
from .Glow import Glow
import torch


class GTGF(nn.Module):
    def __init__(self, args):
        super(GTGF, self).__init__()
        self.args = args
        self.K = args.K
        self.obs_length = args.obs_length

        self.temporal_spatial_transformer = TS_transformer(args)
        self.glow = Glow(args)
        self.decoder = Decoder(args)
        self.loss = Loss(args)

    def forward(self, inputs, iftrain):

        fut = inputs[0][self.obs_length:, :, :2].permute(1, 0, 2) #B1
        his_enc, fut_enc = self.temporal_spatial_transformer(inputs)

        his_enc = his_enc.unsqueeze(2).repeat(1, 1, self.K)
        fut_enc = fut_enc.unsqueeze(2).repeat(1, 1, self.K)

        if iftrain:
            Z, log_det = self.glow(fut_enc, his_enc)
            loss_flow = self.loss.glow(Z, log_det)
        else:
            loss_flow = 0
        # predicting
        dec = self.glow.reverse(his_enc)
        pred_traj = self.decoder(dec.permute(0, 2, 1))
        # cal loss
        loss_traj = self.loss.traj(pred_traj, fut)
        loss_dict = {'loss_flow': loss_flow, 'loss_traj': 0.5 * loss_traj}

        return pred_traj, loss_dict


def Traj_loss(pred, target):

    K = pred.shape[2]
    target = target.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=1)
    loss_traj = traj_rmse.mean()

    return loss_traj


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = self.args.device
        self.K = args.K
        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = self.args.obs_length

        self.lstm = nn.LSTM(self.feats_hidden, self.feats_hidden)
        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, self.feats_hidden//2)
        self.pro4 = nn.Linear(self.feats_hidden//2, self.feats_out)
        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        batch = inputs[0] # B2
        tar_y = batch[self.obs_length:, :, :2].permute(1, 0, 2).to(self.device)
        padding = torch.zeros_like(batch)
        batch_in = torch.cat([batch[:self.obs_length, :, :], padding[self.obs_length:, :, :]])

        enc = self.pro1(batch_in)
        enc = self.relu(enc)
        enc = self.pro2(enc)
        out, _ = self.lstm(enc)
        out = self.pro3(out)
        tra = self.pro4(out) # b2
        # b1
        traj = tra[self.obs_length:, :, :].permute(1, 0, 2).unsqueeze(2).repeat(1, 1, 20, 1)

        traj_loss = Traj_loss(traj, tar_y)
        glow_loss = torch.tensor([0]).to(self.device)
        loss_dict = {'loss_flow': glow_loss, 'loss_traj': traj_loss}

        return traj, loss_dict

