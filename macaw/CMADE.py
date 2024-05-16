import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.register_buffer('bias_mask', torch.ones(out_features))

    def set_mask(self, mask, bias_mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        self.bias_mask.data.copy_(torch.from_numpy(bias_mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias_mask * self.bias)


class CMADE(nn.Module):
    """
    Adapted from an implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
    """

    def __init__(self, nin, nout, edges, h_multiple):
        """
        nin: integer; number of inputs
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        edges: edges of the predefined causal graph variables
        h_multiple: list of numbers of hidden units as multiples of effect variables
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        nhidden = [nin * h for h in h_multiple]

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + nhidden + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.net = nn.Sequential(*self.net)

        # Create the causal graph
        G = nx.DiGraph()
        nodes = np.arange(nin).tolist()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # identify the parents of each node
        parents = [list(G.predecessors(n)) for n in G.nodes]

        # construct the mask matrices for hidden layers
        masks = []
        bias_masks = []
        for l in range(len(nhidden)):
            bias_mask = np.zeros(nin) > 0
            mask = np.zeros((nin, nin)) > 0
            for i in range(nin):
                mask[parents[i] + [i], i] = True

                if len(parents[i]):
                    bias_mask[i] = True

            bias_mask = np.hstack([bias_mask] * h_multiple[l])
            bias_masks.append(bias_mask)

            if l == 0:
                mask = np.hstack([mask] * h_multiple[l])
            else:
                mask = np.vstack([np.hstack([mask] * h_multiple[l])] * h_multiple[l - 1])
            masks.append(mask)

        # construct the mask matrices for output layer
        k = int(nout / nin)
        mask = np.zeros((nin, nin)) > 0
        bias_mask = np.zeros(nin) > 0
        for i in range(nin):
            mask[parents[i], i] = True
            if len(parents[i]):
                bias_mask[i] = True

        mask = np.vstack([np.hstack([mask] * k)] * h_multiple[-1])
        masks.append(mask)

        bias_mask = np.hstack([np.hstack([bias_mask] * k)])
        bias_masks.append(bias_mask)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m, bm in zip(layers, masks, bias_masks):
            l.set_mask(m, bm)

    def forward(self, x):
        return self.net(x)
