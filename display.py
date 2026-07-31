import json
import pandas as pd
import plotly
import numpy as np

def flatten_npvlg_indexed(npvlg_indexed):
    """
    Flatten npvlg_indexed data structure into a list of dictionaries.
    """
    blocks = []
    for key in npvlg_indexed.keys():
        npvlg_indexed[key] = {int(k): v for k, v in npvlg_indexed[key].items()}
        for node in npvlg_indexed[key].keys():
            npvlg_indexed[key][node] = {int(k): v for k, v in npvlg_indexed[key][node].items()}
            for sub_period in npvlg_indexed[key][node].keys():
                added = npvlg_indexed[key][node][sub_period]
                added['year']   = key
                added['sub_period'] = sub_period
                added['block'] = node
                blocks.append(added)
    return blocks

with open('./inputs/npvlg_indexed.json', 'r') as f:
    npvlg_indexed = json.load(f)

blocks = flatten_npvlg_indexed(npvlg_indexed)
blocks = pd.DataFrame(blocks)

# Filter for tonnage_fraction > 0
blocks = blocks[blocks['tonnage_fraction'] > 0.0].copy()

# --- Infer block dimensions ---
def infer_inc(vals: np.ndarray) -> float:
    u = np.unique(vals)
    if len(u) < 2:
        return 1.0 # Fallback
    diffs = np.diff(np.sort(u))
    diffs = diffs[diffs > 1e-6] # Drop zero/tiny diffs
    return float(np.median(diffs)) if len(diffs) > 0 else 1.0

x_inc = infer_inc(blocks['x_c'].to_numpy())
y_inc = infer_inc(blocks['y_c'].to_numpy())
z_inc = infer_inc(blocks['z_c'].to_numpy())

fig = plotly.graph_objs.Figure()

# Get unique values for sliders
years = sorted(blocks['year'].unique())
sub_periods = sorted(blocks['sub_period'].unique())

# Create a trace for each combination of year and sub_period
for year in years:
    for sub_period in sub_periods:
        df_filtered = blocks[(blocks['year'] == year) & (blocks['sub_period'] == sub_period)]
        if not df_filtered.empty:
            # --- Generate vertices and faces for Mesh3d ---
            all_x, all_y, all_z = [], [], []
            all_i, all_j, all_k = [], [], []
            all_intensity = []
            
            # Offsets from center to corners
            dx, dy, dz = x_inc / 2, y_inc / 2, z_inc / 2

            for idx, row in enumerate(df_filtered.itertuples()):
                xc, yc, zc = row.x_c, row.y_c, row.z_c
                
                # Append 8 vertices for the current block
                all_x.extend([xc-dx, xc+dx, xc+dx, xc-dx, xc-dx, xc+dx, xc+dx, xc-dx])
                all_y.extend([yc-dy, yc-dy, yc+dy, yc+dy, yc-dy, yc-dy, yc+dy, yc+dy])
                all_z.extend([zc-dz, zc-dz, zc-dz, zc-dz, zc+dz, zc+dz, zc+dz, zc+dz])
                
                # Use sub_period for color intensity on each vertex
                all_intensity.extend([row.sub_period] * 8)

                # Append 12 faces (triangles) for the current block
                offset = idx * 8
                all_i.extend([offset+0, offset+0, offset+0, offset+0, offset+1, offset+1, offset+2, offset+2, offset+3, offset+3, offset+4, offset+5])
                all_j.extend([offset+1, offset+3, offset+4, offset+2, offset+2, offset+5, offset+6, offset+7, offset+4, offset+7, offset+7, offset+6])
                all_k.extend([offset+2, offset+4, offset+5, offset+3, offset+6, offset+6, offset+3, offset+4, offset+0, offset+0, offset+6, offset+7])

            fig.add_trace(
                plotly.graph_objs.Mesh3d(
                    x=all_x,
                    y=all_y,
                    z=all_z,
                    i=all_i,
                    j=all_j,
                    k=all_k,
                    intensity=all_intensity,
                    colorscale='jet',
                    cmin=min(sub_periods) if sub_periods else 0,
                    cmax=max(sub_periods) if sub_periods else 1,
                    customdata=df_filtered[['block', 'year', 'sub_period']],
                    hovertemplate='<b>Block</b>: %{customdata[0]}<br><b>Year</b>: %{customdata[1]}<br><b>Sub Period</b>: %{customdata[2]}<extra></extra>',
                    name=f'Y:{year}, SP:{sub_period}',
                    # Initially, only show the first year and first sub-period
                    visible=(year == years[0] and sub_period == sub_periods[0]),
                    # Link to the single color bar
                    coloraxis="coloraxis1",
                )
            )

# --- Create Sliders ---
sliders = [
    # Year Slider
    dict(
        active=0,
        y=0.1,  # Position this slider slightly up
        currentvalue={"prefix": "Year: "},
        pad={"t": 20, "b": 10},
        steps=[dict(label=str(year), method='update', args=[{'visible': [tr.name.startswith(f'Y:{year}') for tr in fig.data]}]) for year in years]
    ),
    # Sub-period Slider
    dict(
        active=0,
        y=0,  # Position this slider at the bottom
        currentvalue={"prefix": "Sub Period: "},
        pad={"t": 20, "b": 10},
        steps=[dict(label=str(sp), method='update', args=[{'visible': [tr.name.endswith(f'SP:{sp}') for tr in fig.data]}]) for sp in sub_periods]
    )
]

fig.update_layout(
    sliders=sliders,
    scene=dict(
        xaxis=dict(title='x_c', range=[blocks['x_c'].min(), blocks['x_c'].max()]),
        yaxis=dict(title='y_c', range=[blocks['y_c'].min(), blocks['y_c'].max()]),
        zaxis=dict(title='z_c', range=[blocks['z_c'].min(), blocks['z_c'].max()])
    ),
    # Add a single color bar for all traces
    coloraxis1=dict(
        colorscale='jet',
        cmin=min(sub_periods) if sub_periods else 0,
        cmax=max(sub_periods) if sub_periods else 1,
        colorbar=dict(title='Sub Period')
    ),
    showlegend=False
)

fig.show()
