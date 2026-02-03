import torch 
from torch.nn.functional import softplus

def equality_constraint(x,target):
    return ((x-target)**2).mean()
def binary_equality_log_column(x,tau=1.0):
    x = x.clamp_min(1e-10)
    return -(tau*(x*torch.log(x)).sum(dim=-1)).sum()
def binary_equality_log_row(x,tau=1.0):
    x = x.clamp_min(1e-10)
    x = x / (x.sum(dim=0, keepdim=True) + 1e-10)
    return -(tau*(x*torch.log(x)).sum(dim=0)).sum()
def less_than_inequality_constraint(x):
    return softplus(x).sum()
def greater_than_inequality_constraint(x):
    return softplus(-x).sum()
def constraint_sum_less_than_1(x):
    return x-1