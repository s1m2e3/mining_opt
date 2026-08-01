from problems import PushbackScheduleMIP
from lg_utils import load_pushback_schedule_data
import json
import time

# Columns spaced far enough apart (and levels capped at 1) that their
# pushback footprints don't all collapse into a single dominant selection -
# a deep (level 4) pushback's shallow cap can span 100+ units, so testing
# multi-pushback sequencing needs either shallow levels or widely spaced
# columns. Here: every 8th x-column, 3 widely spaced y-rows, level<=1.
xs_all = [109.0, 129.0, 149.0, 169.0, 189.0, 209.0, 229.0, 249.0, 269.0, 289.0,
          309.0, 329.0, 349.0, 369.0, 389.0]
xs = xs_all[::4]
ys = (4.0, 84.0, 164.0)
columns = [(x, y) for x in xs for y in ys]

data = load_pushback_schedule_data(
    'inputs/pushbacks.csv',
    num_periods=2,
    period_size=1,
    discount_rate=0.1,
    columns=columns,
    max_level=1,
    # binding capacity (grid-block count proxy) so periods 0 and 1 both
    # get used instead of everything landing in period 0 under discounting
    block_count_capacity_per_period=80,
)

model = PushbackScheduleMIP(data)
now = time.time()
model.writeModel()
solution_data = model.solve()
print(time.time()-now)
with open('./inputs/pushback_schedule_smoketest.json', 'w') as f:
    json.dump(solution_data, f, indent=4)
