
import pandas as pd
import json 
import numpy as np

class Simulator:
    def __init__(self, pushbacks, pushback_blocks, blocks):
        # Type casting
        float_cols = ['x', 'y', 'z', 'income', 'cost', 'income_look_ahead', 'cost_look_ahead']
        pushbacks[float_cols] = pushbacks[float_cols].astype(float)
        pushbacks[['enabled', 'mined']] = pushbacks[['enabled', 'mined']].astype(bool)
        
        # OPTIMIZATION: Pre-parse 'blocks' strings into sets once during init
        if isinstance(pushbacks['blocks'].iloc[0], str):
            pushbacks['blocks'] = pushbacks['blocks'].apply(
                lambda x: set(map(int, x.strip('[]').split(',')))
            )
        
        self.original_pushbacks = pushbacks
        self.original_pushback_blocks = pushback_blocks
        self.original_blocks = blocks
        
        # OPTIMIZATION: Use a set for O(1) lookups
        self.mined_blocks = set()

    def _update_neighbor_states(self, pushbacks, loc_x, loc_y, loc_z):
        """Enables the pushback directly above the mined location."""
        mask = (pushbacks['x'] == loc_x) & \
               (pushbacks['y'] == loc_y) & \
               (pushbacks['z'] == loc_z - 20.0)
        pushbacks.loc[mask, 'enabled'] = True
        return pushbacks

    def process_pushbacks(self, pushbacks, chosen_indices):
        if not chosen_indices:
            return pushbacks

        # 1. Extract chosen data and mark as mined
        chosen_df = pushbacks.loc[chosen_indices].copy()
        pushbacks.loc[chosen_indices, 'mined'] = True

        # 2. Update mined_blocks set and enable vertical neighbors
        for _, row in chosen_df.iterrows():
            self.mined_blocks.update(row['blocks'])
            pushbacks = self._update_neighbor_states(pushbacks, row['x'], row['y'], row['z'])

        # 3. Handle 'related' pushbacks (Proportion updates)
        # We only care about enabled, non-mined pushbacks in the vicinity
        # Using a spatial bounding box approach
        mined_coords = chosen_df[['x', 'y', 'z', 'level']].drop_duplicates()
        
        for _, m_row in mined_coords.iterrows():
            # Filter for pushbacks within the 100-unit window
            # Note: Optimized to filter the dataframe fewer times
            spatial_mask = (
                (pushbacks['enabled'] == True) & 
                (pushbacks['mined'] == False) &
                (pushbacks['x'].between(m_row['x'] - 100, m_row['x'] + 100)) &
                (pushbacks['y'].between(m_row['y'] - 100, m_row['y'] + 100)) &
                (pushbacks['z'] <= m_row['z'] + 100)
            )
            
            # Identify lower levels at same loc to drop (original logic)
            depth_mask = (
                (pushbacks['x'] == m_row['x']) & 
                (pushbacks['y'] == m_row['y']) & 
                (pushbacks['z'] == m_row['z']) &
                (pushbacks['level'] < m_row['level'])
            )
            pushbacks.drop(pushbacks[depth_mask].index, inplace=True, errors='ignore')

            # Update proportions for blocks remaining in the spatial window
            related_indices = pushbacks.loc[spatial_mask].index
            for idx in related_indices:
                block_set = pushbacks.at[idx, 'blocks']
                total_count = len(block_set)
                if total_count == 0: continue
                
                # Intersection of sets is very fast
                blocks_already_mined = len(block_set.intersection(self.mined_blocks))
                
                if blocks_already_mined > 0:
                    proportion_remaining = 1.0 - (blocks_already_mined / total_count)
                    
                    if proportion_remaining <= 0:
                        pushbacks.drop(idx, inplace=True)
                    else:
                        # Update economic values based on what's left
                        pushbacks.at[idx, 'income'] *= proportion_remaining
                        pushbacks.at[idx, 'cost'] *= proportion_remaining
        # 4. Final Cleanup
        return pushbacks[pushbacks['mined'] == False].reset_index(drop=True)

    def sample_pushbacks(self, pushbacks, K=200):
        sub_df = pushbacks[(pushbacks['mined'] == False)]
        print(sub_df.shape)
        condition = True if sub_df.shape[0]<K else False 
        if condition:
            return sub_df,condition
        max_level = sub_df['level'].max()
        level_ones = K // (max_level+1)
        # Guard against sampling more than available
        min_level = sub_df['level'].min()
        n_to_sample = min(level_ones, sub_df.loc[sub_df['level']==min_level].shape[0])
        sub_df = sub_df.loc[sub_df['level']==min_level].sample(n=n_to_sample, random_state=42)
        print(sub_df.shape)
        for x,y,z in zip(sub_df['x'],sub_df['y'],sub_df['z']):
            mask = (pushbacks['x'] == x) & (pushbacks['y'] == y) & (pushbacks['z'] == z) & (pushbacks['level']>0)
            sub_df = pd.concat([sub_df, pushbacks.loc[mask]])
        sub_df = self.add_null(sub_df)
        return sub_df,condition
    def add_null(self, pushbacks):
    # row values must be in the same order as pushbacks.columns
        new_row = pd.DataFrame(
        [[0, 0, 0, 0, 0, 0, 0, 1, True, False, []]],
        columns=pushbacks.columns
        )
        return pd.concat([pushbacks, new_row], ignore_index=True)

    def reset(self):
        self.mined_blocks = set()
