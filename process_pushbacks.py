import pandas as pd
import numpy as np
import json

def process_pushbacks(df,levels_down=5,look_ahead_levels=5,extra_gap=5,cost_per_level=-5000,cost_per_block=-1000):
    z_max = df['z'].max()
    z_min = df['z'].min()
    pushbacks = []
    pushback_blocks = {}
    blocks_pushbacks = {}
    for index, row in df.iterrows():
        pushback_blocks[index]={}
        if index % 10 == 0: print(index)
        for j in range(levels_down):
            horizontal_expansion = [k for k in range(j+1, -1, -1)]
            vertical_expansion = list(reversed(horizontal_expansion))
            blocks = []
            income = 0
            cost = 0
            enabled = True if row['z']==z_max else False
            mined = False
            for k in range(len(horizontal_expansion)):
                x_max_right = row['x'] + horizontal_expansion[k]*row['x_step']+extra_gap
                x_max_left = row['x'] - horizontal_expansion[k]*row['x_step']-extra_gap
                y_max_up = row['y'] + horizontal_expansion[k]*row['y_step']+extra_gap
                y_max_down = row['y'] - horizontal_expansion[k]*row['y_step']-extra_gap
                z_max_level = row['z'] - vertical_expansion[k]*row['z_step']
                sub_df_level = df.loc[(df['x']>=x_max_left) & (df['x']<x_max_right) & (df['y']>=y_max_down) & (df['y']<y_max_up) & (df['z']==z_max_level)]
                income += sub_df_level['income'].sum()
                cost += cost_per_block*sub_df_level.shape[0]+cost_per_level*vertical_expansion[k]*sub_df_level.shape[0]
                blocks.extend(sub_df_level['index'].tolist())
            look_ahead_x_max_right = row['x'] + horizontal_expansion[0]*row['x_step']+extra_gap
            look_ahead_x_max_left = row['x'] - horizontal_expansion[0]*row['x_step']-extra_gap 
            look_ahead_y_max_up = row['y'] + horizontal_expansion[0]*row['y_step']+extra_gap
            look_ahead_y_max_down = row['y'] - horizontal_expansion[0]*row['y_step']-extra_gap
            look_ahead_z_max_level = row['z'] - vertical_expansion[-1]*row['z_step']+extra_gap
            look_ahead_z_min_level = row['z'] - (vertical_expansion[-1]+look_ahead_levels)*row['z_step']-extra_gap
            sub_df_level=df.loc[(df['x']>=look_ahead_x_max_left) & (df['x']<look_ahead_x_max_right) & (df['y']>=look_ahead_y_max_down) & (df['y']<look_ahead_y_max_up) & (df['z']>=look_ahead_z_min_level) & (df['z']<=look_ahead_z_max_level)]
            income_look_ahead = sub_df_level['income'].sum()
            cost_look_ahead = cost_per_block*sub_df_level.shape[0]+cost_per_level*vertical_expansion[k]*sub_df_level.shape[0]
            pushbacks.append([row['x'],row['y'],row['z'],cost,cost_look_ahead,income,income_look_ahead,j,enabled,mined,blocks])
            pushback_blocks[index][j] = blocks
    return pushbacks,pushback_blocks

def main():
    df = pd.read_csv('inputs/blocks.csv')
    df['z']=-df['z']
    pushbacks, pushback_blocks = process_pushbacks(df)
    pushbacks = pd.DataFrame(pushbacks,columns=['x','y','z','cost','cost_look_ahead','income','income_look_ahead','level','enabled','mined','blocks'])
    pushbacks.to_csv('inputs/pushbacks.csv',index=False)
    with open('inputs/pushback_blocks.json', 'w') as f:
        json.dump(pushback_blocks, f, indent=4)
if __name__ == '__main__':
    main()