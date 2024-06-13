import numpy as np
import torch
import torch.nn as nn

from .CMADE import CMADE


class Flow(nn.Module):
    """ Masked Causal Flow that uses a MADE-style network for fast-forward """

    def __init__(self, dim, edges, device, net_class=CMADE, hm=[4, 6, 4]):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim * 2, edges, hm)
        self.device = device

    def forward(self, x):
        # # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        # st = self.net(x)
        # s, t = st.split(self.dim, dim=1)
        # s = torch.nan_to_num(s, nan=0.0, posinf=10, neginf=-10)
        # t = torch.nan_to_num(t, nan=0.0, posinf=1000, neginf=-1000)
        #
        # z = x * torch.exp(s) + t
        #
        # log_det = torch.sum(s, dim=1)
        # return z, log_det
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = torch.nan_to_num(self.net(x), nan=0.0, posinf=1e3, neginf=-1e3)
        s, t = st.split(self.dim, dim=1)

        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z).to(self.device)
        log_det = torch.zeros(z.size(0)).to(self.device)
        for i in range(self.dim):
            # st = self.net(x)
            # s, t = st.split(self.dim, dim=1)
            # s = torch.nan_to_num(s, nan=0.0, posinf=10, neginf=-10)
            # t = torch.nan_to_num(t, nan=0.0, posinf=1000, neginf=-1000)
            #
            # x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            # log_det += -s[:, i]
            st = torch.nan_to_num(self.net(x), nan=0.0, posinf=1e3, neginf=-1e3)
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, priors, flows):
        super().__init__()
        self.priors = priors
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        if type(self.priors) == list:
            prior_log_prob = 0
            for sl, dist in self.priors:
                data = zs[-1][:, sl]
                prior_log_prob += dist.log_prob(data).view(x.size(0), -1).sum(1)
        else:
            prior_log_prob = self.priors.log_prob(zs[-1]).view(x.size(0), -1).sum(1)

        return zs, prior_log_prob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        z = self.priors.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs

    def log_likelihood(self, x):
        if type(x) is np.ndarray:
            x = torch.tensor(x.astype(np.float32))
        _, prior_logprob, log_det = self.forward(x)
        return (prior_logprob + log_det).cpu().detach().numpy()
