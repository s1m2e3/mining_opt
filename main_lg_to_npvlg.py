from problems import NPVLG,NPVLG_Indexed, NPVLG_Hierarchical, LowerScheduleMIP
from data_classes import NPVLGData
from lg_utils import LG_to_NPVLG
import json
import time

with open('./inputs/lg_solution.json', 'r') as f:
    data = json.load(f)


data = LG_to_NPVLG(data) 

npvlg_data = NPVLGData(
    blocks=data['blocks'],
    num_periods=2,
    period_size=1,
    time_penalty=1, # Per period
    ramp=10, # Defined in periods
    daily_mining_rate=1e4,
    discount_rate=0.1
)

model = LowerScheduleMIP(npvlg_data.__dict__)
now = time.time()
model.writeModel()
solution_data = model.solve()
print(time.time()-now)
with open('./inputs/npvlg_indexed.json', 'w') as f:
    json.dump(solution_data, f, indent=4)
