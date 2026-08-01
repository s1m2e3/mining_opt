from problems import PushbackScheduleMIP
from lg_utils import load_pushback_schedule_data
import json
import time

data = load_pushback_schedule_data(
    'inputs/pushbacks.csv',
    num_periods=2,
    period_size=1,
    discount_rate=0.1,
)

model = PushbackScheduleMIP(data)
now = time.time()
model.writeModel()
solution_data = model.solve()
print(time.time()-now)
with open('./inputs/pushback_schedule.json', 'w') as f:
    json.dump(solution_data, f, indent=4)
