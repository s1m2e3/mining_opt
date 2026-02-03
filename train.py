import pandas as pd
import json
from simulator import Simulator
from model import SimpleTransformer
import torch 
from utils import income_loss,entropy_loss,income, constraint_sequencing, npv
import numpy as np
import os
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pushbacks = pd.read_csv('inputs/pushbacks.csv')
pushback_blocks = json.load(open('inputs/pushback_blocks.json'))
blocks = pd.read_csv('inputs/blocks.csv')

dim_blocks = 10
d_model = 64*4
K_eval = [150,100,50]
num_iterations_epoch = 10
num_layers = 20
epochs = 5
B=1

simulator = Simulator(pushbacks=pushbacks,pushback_blocks=pushback_blocks,blocks=blocks)
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
proportions = {}
for k_eval in K_eval:
    model = SimpleTransformer(in_dim=8,d_model=d_model,out_dim=1, num_layers=num_layers,use_posenc=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    K_total = k_eval
    K_sub = k_eval
    proportions[k_eval] = {}
    for k in range(epochs):
        proportions[k_eval][k] = {}
        pushbacks = simulator.original_pushbacks.copy()
        for i in range(num_iterations_epoch):
            proportions[k_eval][k][i] = {}
            optimizer.zero_grad()
            sampled_pushbacks=[]
            sampled_df = []
            for j in range(B):
                df,condition = simulator.sample_pushbacks(pushbacks,K=K_total)
                sampled_df.append(df)
                sampled_pushbacks.append(torch.from_numpy(sampled_df[-1].to_numpy()[:,:-1].astype('float32')))
            if condition: 
                break
            x_unnormalized = torch.stack(sampled_pushbacks).to(device)
            x_normalized = (x_unnormalized - x_unnormalized.min(dim=-1,keepdim=True)[0]) / (x_unnormalized.max(dim=-1,keepdim=True)[0] - x_unnormalized.min(dim=-1,keepdim=True)[0])
            t_nn = time.time()
            y_pred = torch.softmax(model(x_normalized[:,:,:-2]).squeeze(-1),dim=-1)
            y_pred_constrained = constraint_sequencing(model,x_normalized[:,:,:-2],x_unnormalized).squeeze(-1)
            t_nn = time.time()-t_nn
            chosen_values, chosen_indices = torch.topk(y_pred_constrained,k=min(K_sub+1,y_pred_constrained.shape[1]),dim=1)
            t_greedy = time.time()
            out_greedy = income(y_pred_constrained, sampled_df, discount_factor=0.99, steps_to_ahead=8,beam_width=1,rollout_horizon=1,candidate_pool_size=1)
            t_greedy = time.time()-t_greedy
            t_k_step = time.time()
            out_k_step = income(y_pred_constrained, sampled_df, discount_factor=0.99, steps_to_ahead=8,beam_width=4,rollout_horizon=k_eval//5,candidate_pool_size=k_eval//20)
            t_k_step = time.time()-t_k_step
            weighted_npv,npv_tensor = (income_loss(y_pred_constrained, sampled_df))
            nn_unnormalized = torch.gather(x_unnormalized,dim=1,index=chosen_indices.unsqueeze(-1).expand(-1, -1, x_unnormalized.size(-1)) )
            greedy_unnormalized = torch.gather(x_unnormalized, dim=1,index=out_greedy['orders'].unsqueeze(-1).expand(-1, -1, x_unnormalized.size(-1)))
            k_step_unnormalized = torch.gather(x_unnormalized, dim=1,index=out_k_step['orders'].unsqueeze(-1).expand(-1, -1, x_unnormalized.size(-1)))
            nn_npv = npv(nn_unnormalized)/1e6
            heuristic_npv_greedy = npv(greedy_unnormalized)/1e6
            heuristic_npv_k_step = npv(k_step_unnormalized)/1e6
            entropy = (entropy_loss(y_pred_constrained,1e-10))
            kl_divergence = kl_loss(torch.log(y_pred),y_pred_constrained)*1e1
            print(i, nn_npv.item(),heuristic_npv_greedy.item(),heuristic_npv_k_step.item(),entropy.item(),(kl_divergence*0.1).item())
            proportions[k_eval][k][i]['nn_value'] = nn_npv.item()
            proportions[k_eval][k][i]['heuristic_greedy_value'] = heuristic_npv_greedy.item()
            proportions[k_eval][k][i]['heuristic_k_step_value'] = heuristic_npv_k_step.item()
            proportions[k_eval][k][i]['nn_time'] = t_nn
            proportions[k_eval][k][i]['heuristic_greedy_time'] = t_greedy
            proportions[k_eval][k][i]['heuristic_k_step_time'] = t_k_step
            proportions[k_eval][k][i]['entropy'] = (entropy*1e-3).item()
            proportions[k_eval][k][i]['kl_divergence'] = (kl_divergence/1e1).item()
            total = -weighted_npv+entropy
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
            norm_grad = 0
            for p in model.parameters():
                if p.grad is not None:
                    norm_grad += p.grad.norm().item()**2
            norm_grad = np.sqrt(norm_grad)
            print(f'norm of the gradients: {norm_grad:.4f}')
            optimizer.step()
            chosen_pushbacks = chosen_indices[0].detach().cpu().numpy().tolist()
            pushbacks = simulator.process_pushbacks(pushbacks,chosen_pushbacks)
            
        simulator.reset()
with open(os.path.join('outputs', f'loss.json'), 'w') as f:
    json.dump(proportions, f, indent=4)