import torch

def gradient_step(f,x,lr=0.1):
    grad = torch.autograd.grad(f(x), x, create_graph=True,retain_graph=True)[0]
    return x - lr * grad
def conditional_optim(f1_target,f2_target,x,f,f_constraint,inequality_constraint):
    eval = torch.where(f_constraint(f(x))>0.1)[0]
    chosen_blocks = torch.argmax(f(x),dim=1).long()
    counts = torch.bincount(chosen_blocks)
    non_one = (counts > 1).nonzero(as_tuple=True)[0] 
    if eval.shape[0]>0 and non_one.shape[0]>0:
        index = torch.where(f_constraint(f(x))>0.1)[0]
        return lambda x: f1_target(f(x)) + f2_target(f(x)) + inequality_constraint(f_constraint(f(x))[index])
    elif non_one.shape[0]>0:
        return lambda x: f1_target(f(x)) + f2_target(f(x))
    elif eval.shape[0]>0:
        index = torch.where(f_constraint(f(x))>0.1)[0]
        return lambda x: inequality_constraint(f_constraint(f(x))[index])
    
def optimize(x,f,f_target_1,f_target_2,f_constraint,inequality_constraint,max_steps=100,proportion=0.9):
    f_constraint_masked = torch.where(f_constraint(f(x))>0,f_constraint(f(x)),0)
    optimize_constraint = conditional_optim(f_target_1,f_target_2,x,f,f_constraint,inequality_constraint)
    counter = 0
    eval = torch.where(f_constraint(f(x))>0.1)[0]
    chosen_blocks = torch.argmax(f(x),dim=1).long()
    counts = torch.bincount(chosen_blocks)
    non_one = (counts > 1).nonzero(as_tuple=True)[0]
    while counter < max_steps and non_one.shape[0]<x.shape[0]*proportion:
        x = gradient_step(optimize_constraint,x)
        optimize_constraint = conditional_optim(f_target_1,f_target_2,x,f,f_constraint,inequality_constraint)
        counter += 1
        chosen_blocks = torch.argmax(f(x),dim=1).long()
        counts = torch.bincount(chosen_blocks)
        non_one = (counts > 1).nonzero(as_tuple=True)[0]
    return x