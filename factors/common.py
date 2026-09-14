"""Named quadratic/hinge kernels; normalization is explicit at each call site."""

def mean_square(value, sigma=1., weight=1.):
    return weight * (value / sigma).square().mean()


def magnitude_hinge(value, onset, weight=1.):
    import torch
    return weight * torch.relu(torch.linalg.vector_norm(value, dim=-1) - onset).square().mean()
