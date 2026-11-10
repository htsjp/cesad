import torch


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def grad_reverse(x, weight=1.0):
    return GradientReverse.apply(x, weight)
