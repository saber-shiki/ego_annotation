"""Explicit row-order temporal differences; no implicit timestamp conversion."""
import torch


def parameter_velocity(value, weight):
    if len(value)<2:return value.new_zeros(())
    return weight*(value[1:]-value[:-1]).square().mean()


def parameter_acceleration(value, weight):
    if len(value)<3:return value.new_zeros(())
    return weight*(value[2:]-2*value[1:-1]+value[:-2]).square().mean()


def correction_velocity(current,previous,gap,sigma):
    import numpy as np
    return (current-previous)/(sigma*np.sqrt(float(gap)))


def correction_acceleration(previous,current,following,gap0,gap1,sigma):
    import numpy as np
    return ((following-current)/gap1-(current-previous)/gap0)/(sigma*np.sqrt(float(max(gap0,gap1))))
