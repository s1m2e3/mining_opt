
from problems import LG
import json
import time
import random

with open('./inputs/block_model_with_dependencies_smoketest.json', 'r') as f:
    data = json.load(f)
data['blocks'] = {int(k): v for k, v in data['blocks'].items()}
for k,v in data['blocks'].items():
    v['value']=v['value'] if random.random() > 0.5 else -v['value']

model = LG(data)
now = time.time()
model.writeModel()
print(time.time()-now)
now = time.time()
solution_data = model.solve()
print(time.time()-now)
with open('./inputs/lg_solution_smoketest.json', 'w') as f:
    json.dump(solution_data, f, indent=4)
