"""Explicit row-order temporal differences; no implicit timestamp conversion."""
import torch


def parameter_velocity(value, weight):
    if len(value)<2:return value.new_zeros(())
    return weight*(value[1:]-value[:-1]).square().mean()


def parameter_acceleration(value, weight):
    if len(value)<3:return value.new_zeros(())
    return weight*(value[2:]-2*value[1:-1]+value[:-2]).square().mean()
