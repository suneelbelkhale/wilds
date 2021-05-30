from typing import Optional, Mapping, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x


class BranchedModules(nn.ModuleDict):
    def __init__(self, order=List[str], modules: Optional[Mapping[str, nn.Module]] = None, cat_dim=None) -> None:
        super(BranchedModules, self).__init__(modules)
        assert modules is not None
        self._order = order
        self._cat_dim = cat_dim
        assert set(order) == set(modules.keys()), [order, modules.keys()]

    def forward(self, input_, ret_dict=False):
        all = dict()
        all_ls = []
        for k in self._order:
            all[k] = self[k](input_)
            all_ls.append(all[k])

        if ret_dict:
            return all

        if self._cat_dim is not None:
            return torch.cat(all_ls, dim=self._cat_dim)

        return all_ls


class RevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGradLayer(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGrad.apply(input_, self._alpha)