import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

N = 4

def harmonic(x,y):
    return 2/( (1/x) + (1/y) )

ranks = np.zeros((N+1,N+1))
for r1 in np.arange(1, N+1, 1):
    for r2 in np.arange(1, N+1, 1):
        ranks[r1,r2] = harmonic(r1,r2)
sns.heatmap(ranks)
plt.savefig("harmonic.png")
