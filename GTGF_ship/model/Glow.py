
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import nflows.utils.typechecks as check


class Invertible1x1Conv(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.randn(c, c))[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z):
        # shape
        B, F, K, = z.shape
        W = self.conv.weight.squeeze()
        log_det_W = B * K * torch.logdet(W)
        z = self.conv(z)
        return z, log_det_W

    def reverse (self, z):

        W = self.conv.weight.squeeze()

        if not hasattr(self, 'W_inverse'):
            # Reverse computation
            bias = torch.diag(torch.zeros_like(W[0])+1e-6)
            W_inverse = (W+bias).float().inverse()
            W_inverse = Variable(W_inverse[..., None])
            if z.type() == 'torch.cuda.HalfTensor':
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse
        z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        return z



class GroupInstanceNorm(nn.Module):
    def __init__(self, features=None, group_len=5):
        """
        Args:
            group_len: the len of group
            tensor: shape of (B, L, K), K for the number of trajectory
        Returns: shape of (B, L, K)
        """
        if not check.is_positive_int(group_len):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()
        self.group_len = group_len
        self.features = features
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        # self.scale = nn.Parameter(torch.zeros(group_len))
        if features is not None:
            self.log_scale = nn.Parameter(torch.zeros(features, group_len))
            self.shift = nn.Parameter(torch.zeros(features, group_len))
        else:
            self.log_scale = nn.Parameter(torch.zeros(group_len))
            self.shift = nn.Parameter(torch.zeros(group_len))

    @property
    def scale(self):
        return torch.exp(torch.clamp(self.log_scale, max=1e10))

    def forward(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("Expecting inputs to be a 3D tensor.")
        if inputs.dim() == 3:
            K_sample = inputs.shape[2]
            if K_sample % self.group_len != 0:
                raise ValueError("Expecting K_sample to be divisible by group_len.")
            B, C, K = inputs.shape
            if self.features is not None:
                inputs = inputs.unfold(2, self.group_len, self.group_len)
            else:
                inputs = inputs.reshape(-1, K)
                inputs = inputs.unfold(1, self.group_len, self.group_len)

        if self.training and not self.initialized:
            self._initialize(inputs)

        if self.features is not None:
            scale = self.scale.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            shift = self.shift.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            outputs = scale * inputs + shift
            outputs = outputs.reshape(B, C, -1)
        else:
            scale = self.scale.view(1, -1).repeat(inputs.shape[1], 1)
            shift = self.shift.view(1, -1).repeat(inputs.shape[1], 1)
            outputs = scale * inputs + shift
            outputs = outputs.reshape(-1, K).reshape(-1, C, K)

        log_det = torch.sum(self.log_scale)

        return outputs, log_det

    def inverse(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("Expecting inputs to be a 3D tensor.")
        if inputs.dim() == 3:
            K_sample = inputs.shape[2]
            if K_sample % self.group_len != 0:
                raise ValueError("Expecting K_sample to be divisible by group_len.")
            B, C, K = inputs.shape
            if self.features is not None:
                inputs = inputs.unfold(2, self.group_len, self.group_len)
            else:
                inputs = inputs.reshape(-1, K)
                inputs = inputs.unfold(1, self.group_len, self.group_len)

        if self.features is not None:
            scale = self.scale.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            shift = self.shift.view(C, 1, -1).repeat(1, inputs.shape[2], 1)
            outputs = (inputs - shift) / (scale + 1e-6)
            outputs = outputs.reshape(B, C, -1)
        else:
            scale = self.scale.view(1, -1).repeat(inputs.shape[1], 1)
            shift = self.shift.view(1, -1).repeat(inputs.shape[1], 1)
            outputs = (inputs - shift) / (scale + 1e-6)
            outputs = outputs.reshape(-1, K).reshape(-1, C, K)

        return outputs

    def _initialize(self, inputs):
        """Data-dependent initialization"""
        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / (std + 1e-6)).mean(dim=0)
            std = std.mean(-2)
            mu = mu.mean(-2)
            self.log_scale.data = -torch.log(torch.clamp(std, 1e-10, 1e10))
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)



class Affinecoupling(nn.Module):

    def __init__(self, feats_hidden, feats_block):
        super(Affinecoupling, self).__init__()
        self.z_embed = nn.Sequential(nn.Linear(feats_hidden + feats_block//2, 2 * feats_hidden),
                                     nn.GELU(),
                                     nn.Linear(2 * feats_hidden, 256),
                                     nn.GELU(),)
        for m in self.z_embed.modules():
            if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.05)
                    m.bias.data.zero_()
        end = nn.Linear(256, feats_block)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.tanh = nn.Tanh()
        self.end = end

    def forward(self, z, history_enc):

        z_0, z_1 = z.chunk(2, 1)
        h_x = torch.cat([history_enc, z_0], 1)
        h_x = self.z_embed(h_x.permute(0, 2, 1))
        h_x = self.end(h_x).permute(0, 2, 1)  # 5.22
        h_x = self.tanh(h_x)
        log_s, b = h_x.chunk(2, 1)
        z_1 = torch.exp(log_s) * z_1 + b
        z = torch.cat([z_0, z_1], 1)
        log_det = torch.sum(log_s)

        return z, log_det

    def reverse(self, z_sample, history_enc):

        z_sample_0, z_sample_1 = z_sample.chunk(2, 1)
        h_x = torch.cat([history_enc, z_sample_0], 1)
        h_x = self.z_embed(h_x.permute(0, 2, 1))
        h_x = self.end(h_x).permute(0, 2, 1)  # 5.22
        h_x = self.tanh(h_x)
        s, b = h_x.chunk(2, 1)
        z_sample_1 = (z_sample_1 - b) / torch.exp(s)
        z_sample = torch.cat([z_sample_0, z_sample_1], 1)

        return z_sample

class Glow(nn.Module):
    def __init__(self, args):
        super(Glow, self).__init__()

        self.device = args.device
        self.feats_hidden = args.feats_hidden
        self.blocks = args.blocks
        self.flows = args.flows
        self.K = args.K
        self.feats_base = self.feats_hidden // self.blocks
        self.shift = self.flows // self.blocks

        self.convinv = nn.ModuleList()
        self.instancenorm = nn.ModuleList()
        self.affine = nn.ModuleList()

        feats_block = self.feats_hidden
        for i in range(self.flows):
            if i % self.shift == 0 and i > 0:
                feats_block = feats_block - self.feats_base
            self.instancenorm.append(GroupInstanceNorm(feats_block, group_len= self.K))
            self.convinv.append(Invertible1x1Conv(feats_block))
            self.affine.append(Affinecoupling(self.feats_hidden, feats_block))

    def forward(self, future_enc, history_enc):

        log_det = torch.tensor([0]).to(self.device)
        output_z = []
        for k in range(self.flows):
            if k % self.shift == 0 and k > 0:
                output_z.append(future_enc[:, :self.feats_base, :])
                future_enc = future_enc[:, self.feats_base:, :]

            future_enc, log_det_n = self.instancenorm[k](future_enc)
            future_enc, log_det_c = self.convinv[k](future_enc)
            future_enc, log_det_a = self.affine[k](future_enc, history_enc)

            log_det_ = log_det_n / self.K + log_det_c + log_det_a
            log_det = log_det + log_det_

        output_z.append(future_enc)
        Z = torch.cat(output_z, 1)

        return Z, log_det
    
    def reverse(self, history_enc):
        B, F, K = history_enc.shape
        z_sample = torch.randn(B, self.feats_base, K).to(self.device)
        for k in reversed(range(self.flows)):

            z_sample = self.affine[k].reverse(z_sample, history_enc)
            z_sample = self.convinv[k].reverse(z_sample)
            z_sample = self.instancenorm[k].inverse(z_sample)

            if k % self.shift == 0 and k > 0:
                z = torch.randn(B, self.feats_base, K).to(self.device)
                z_sample = torch.cat((z, z_sample), 1)

        return z_sample
