import pandas as pd
import os
import numpy as np

def createBlocks(grid,df,step):
    lists = []
    for x,y,z in zip(grid[0].flatten(),grid[1].flatten(),grid[2].flatten()):
        x_max = x+step
        y_max = y+step
        z_max = z+step
        mask = (df['XC']>=x) & (df['XC']<x_max) & (df['YC']>=y) & (df['YC']<y_max) & (df['ZC']>=z) & (df['ZC']<z_max)
        df_int = df.loc[mask]
        if df_int.shape[0]==0:
            lists.append([x,y,z,step,step,step,0.0])
        else:
            lists.append([x,y,z,step,step,step,np.round(df['income'].mean(),2)])
    df = pd.DataFrame(lists,columns=['x','y','z','x_step','y_step','z_step','income']).reset_index(drop=False)
    return df

# get the path to the inputs directory
inputs_dir = os.path.join(os.path.dirname(__file__), 'inputs')
step = 20
# import the file "MB_UG.csv" from the inputs directory and mount it in pandas
df = pd.read_csv(os.path.join(inputs_dir, 'MB UG.csv'))
df=df[['XC','YC','ZC','XINC','YINC','ZINC','CU','AU']]
df['vol']=df['XINC']*df['YINC']*df['ZINC']
df['ton']=df['vol']*5
mask = df['AU'].astype('string').str.contains('-', regex=False, na=False)
df.loc[mask,'AU']=0.0
mask = df['CU'].astype('string').str.contains('-', regex=False, na=False)
df.loc[mask,'CU']=0.0
df['income']=df['ton']*(df['AU'].astype('float')/32*4000+df['CU'].astype('float')*0.01*8000)
x_grid = np.arange(int(df['XC'].min()), int(df['XC'].max()), step=step)
y_grid = np.arange(int(df['YC'].min()), int(df['YC'].max()), step=step)
z_grid = np.arange(int(df['ZC'].min()), int(df['ZC'].max()), step=step)
grid = np.meshgrid(x_grid, y_grid, z_grid)
df = createBlocks(grid,df,step)
df.to_csv(os.path.join(inputs_dir, 'blocks.csv'),index=False)
