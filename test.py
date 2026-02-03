import torch
from constraints import less_than_inequality_constraint,\
    binary_equality_log_row,binary_equality_log_column,\
    constraint_sum_less_than_1
from optimization import optimize
from model import SimpleTransformer
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H = 100
N = 200
dim_blocks = 5
dim_global = 3

x = torch.randn(H,N,dim_blocks).requires_grad_(True).to(device)
model = SimpleTransformer(dim_blocks,d_model=16,out_dim=1,use_posenc=False).to(device)
x_out = model(x).squeeze(-1)
y = torch.softmax(x_out,dim=1)
# constraint = lambda x: constraint_sum_less_than_1(x.sum(dim=0))
# inequality_function = less_than_inequality_constraint
# x_out_constrained = optimize(x_out,lambda x: torch.softmax(x,dim=1),binary_equality_log_row,binary_equality_log_column,constraint,inequality_function)
# probs = torch.softmax(x_out_constrained,dim=1)

revenue = income_loss(y)
ent_loss = entropy_loss(y,1e-10)
print(revenue.shape,ent_loss.shape)
# print(choices_k,choices.shape)

# indices = torch.arange(0,H).to(device)
# choices_cut = torch.unique(choices)
# indices = [choices.cpu().numpy().tolist().index(chosen) for chosen in choices_cut.cpu().numpy().tolist()]
# sub_selected_scores = x_out_constrained[torch.tensor(indices),choices[indices]]
# values,indices =  torch.sort(sub_selected_scores,descending=True)
# print(values,indices)

# sub_selected_blocks = x_out_constrained[indices,choices_cut]
# print(sub_selected_blocks.shape)
# print(x_out_constrained.shape)
# y_constrained = torch.softmax(x_out_constrained,dim=1)
# print(y_constrained.sum(1),y_constrained.sum(0))
# print(torch.argmax(y_constrained,dim=1))
