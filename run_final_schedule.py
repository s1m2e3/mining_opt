import json
import time
from lg_utils import load_pushback_schedule_data, beam_search_pushback_schedule

xs_all = sorted(set([109.0 + 20*i for i in range(59)]))
ys_all = [4.0, 24.0, 44.0, 64.0, 84.0, 104.0, 124.0, 144.0, 164.0]

DAILY_MINING_RATE = 10000  # 10k TPD
PERIOD_SIZE = 1  # years, matching data_classes.NPVLGData's convention
NUM_PERIODS = 10
TONNAGE_CAPACITY = DAILY_MINING_RATE * 365 * PERIOD_SIZE
DISCOUNT_RATE = 0.1
BEAM_WIDTH = 40
CANDIDATE_POOL_SIZE = 40

xs = xs_all[:15]
columns = [(x, y) for x in xs for y in ys_all]

t0 = time.time()
data = load_pushback_schedule_data(
    'inputs/pushbacks.csv',
    num_periods=NUM_PERIODS,
    period_size=PERIOD_SIZE,
    discount_rate=DISCOUNT_RATE,
    columns=columns,
    max_level=None,
    tonnage_capacity_per_period=TONNAGE_CAPACITY,
)
prep_time = time.time() - t0
print('prep:', prep_time, 'pushbacks:', len(data['pushbacks']))

t0 = time.time()
sol = beam_search_pushback_schedule(data, beam_width=BEAM_WIDTH, candidate_pool_size=CANDIDATE_POOL_SIZE)
beam_time = time.time() - t0
print('beam:', beam_time, 'chosen:', len(sol['chosen']), 'objective:', sol['objective_value'])

# build a full block-level export for visualization: every raw grid-block
# belonging to a chosen pushback, tagged with the period(s) it was mined in
blocks_df_path = 'inputs/blocks.csv'
import pandas as pd
blocks_lookup = pd.read_csv(blocks_df_path).set_index('index')[['x', 'y', 'z']].to_dict('index')

records = []
for pid in sol['order']:
    v = sol['chosen'][pid]
    schedule = v['schedule']  # {period: fraction}
    n_blocks = len(v['blocks'])
    # allocate blocks to periods in list order, proportional to each
    # period's fraction (matches the pushback's own top-down block order)
    period_items = sorted(schedule.items())
    start_idx = 0
    for period, frac in period_items:
        end_idx = start_idx + round(frac * n_blocks)
        end_idx = min(end_idx, n_blocks)
        for b in v['blocks'][start_idx:end_idx]:
            coord = blocks_lookup.get(b)
            if coord is None:
                continue
            records.append({
                'block_id': b, 'pushback_id': pid, 'level': v['level'],
                'x': coord['x'], 'y': coord['y'], 'z': coord['z'],
                'period': period,
            })
        start_idx = end_idx

export = {
    'objective_value': sol['objective_value'],
    'prep_time': prep_time,
    'beam_time': beam_time,
    'num_pushbacks_candidate': len(data['pushbacks']),
    'chosen_pushback_ids': sol['order'],
    'blocks': records,
}
with open('final_schedule.json', 'w') as f:
    json.dump(export, f)
print('wrote final_schedule.json with', len(records), 'block records across', len(sol['chosen']), 'pushbacks')
