"""Unsigned visual pad proximity has no signed contact authority."""
import torch

def unsigned_pad_loss(delta, weight):
    """Three nearest actual MANO pad samples to full triangle correspondences."""
    return weight * torch.topk(delta.square().sum(-1), 3, largest=False).values.mean() / 0.003**2



def contact_state_prior(probability, prior, strength, active):
    delta=probability-prior
    return (strength[active]*delta[active].square()).sum()/active.sum().clamp_min(1.)


def contact_geometry_likelihood(probability, target, strength, active):
    return contact_state_prior(probability,target,strength,active)


def contact_state_continuity(probability, strength, pairs, temporal_strength):
    i,j=pairs[:,0],pairs[:,1]
    return (.5*(strength[i]+strength[j])*temporal_strength*(probability[j]-probability[i]).square()).mean()


def normal_deadband(normal_gap, deadband, weight, probability):
    # C multiplies geometry loss without detach: both C and geometry receive gradients.
    return weight*probability*torch.relu(torch.abs(normal_gap)-deadband).square().mean()


def normal_scaled_attraction(normal_gap, margin, support_uncertainty, weight, probability):
    precision=(margin.clamp_min(1e-6)/(margin+support_uncertainty).clamp_min(1e-6)).square()
    return weight*probability*precision*normal_gap.square().mean()
