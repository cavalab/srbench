import pandas as pd
import numpy as np
from glob import glob 
from pareto_utils import front

np.random.seed(42)

def bootstrap(val, n = 1000, fn=np.mean):
    val_samples = []
    for i in range(n):
        sample = np.random.randint(0,len(val)-1, size=len(val))
        val_samples.append( fn(val[sample]) )
    m = np.mean(val_samples)
    sd = np.std(val_samples)
    ci_upper  = np.quantile(val_samples,0.95)
    ci_lower  = np.quantile(val_samples,0.05)
    return m, sd, ci_upper,ci_lower

df = pd.read_csv("../docs/csv/blackbox_results.csv")
xcol = 'r2_test'
ycol = 'model_size'
zcol = 'training time (s)'

# outline pareto front
pareto_data = df.groupby('algorithm').median()

def create_front(df, xc, yc):
    objs = df[[xc,yc]].values
    levels = 5

    PFs = []
    for el in range(levels):
        PF = front(-objs[:,0],objs[:,1])
        objs[PF,:] = 1e10
        PFs.append(PF)
        if np.all(objs[:,0]==1e10):
            break

    front_alg = {}
    for i, pf in enumerate(PFs):
        print(f"FRONT {i}")
        for ix in pf:
            name = df.iloc[ix].name 
            if name not in front_alg:
                front_alg[name] = i
                print(f"{name}")
    return front_alg

df['g1']=-1
for k,v in create_front(pareto_data.copy(), xcol, ycol).items():
    df.loc[df.algorithm == k,'g1'] = v
df['g2']=-1
for k,v in create_front(pareto_data.copy(), xcol, zcol).items():
    df.loc[df.algorithm == k,'g2'] = v
df['g3']=-1
for k,v in create_front(pareto_data.copy(), ycol, zcol).items():
    df.loc[df.algorithm == k,'g3'] = v

df.groupby(by="algorithm").median().to_csv("../docs/csv/pareto.csv")
