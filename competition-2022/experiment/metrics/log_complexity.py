import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,2)

for i, ax in enumerate(axs):
  x = np.arange(50) + 1
  log_base = 5 if i == 0 else 10
  y = np.round(np.log(x)/np.log(log_base),1)
  z = rankdata(y, method='dense')
  # higher is better
  z = - z
  # normalize
  z = (z - np.min(z))/(np.max(z)-np.min(z))
  print(log_base, x, z)
  ax.plot(x,z)
  ax.set_xlim(np.min(x),np.max(x))
  ax.set_xlabel("Complexity")
  ax.set_ylabel("Rank")
  ax.set_title("Log base: {}".format(log_base))
  ax.grid(True)
fig.tight_layout()
fig.savefig("example_logcomplexity.png")


fig, ax = plt.subplots(figsize=(5,5))
x = np.arange(100) + 1
log_base = 5
y = np.round(np.log(x)/np.log(log_base),1)
z = rankdata(y, method='dense')
# higher is better
z = - z
# normalize
z = (z - np.min(z))/(np.max(z)-np.min(z))
print(log_base, x, z)
ax.plot(x,z)
ax.set_xlim(np.min(x),np.max(x))
ax.set_xlabel("Number of components")
ax.set_ylabel("Simplicity (normalized for up to {} components)".format(len(x)))
#ax.set_title("Log base: {}".format(log_base))
ax.grid(True)
fig.tight_layout()
fig.savefig("../simplicity_score.png")