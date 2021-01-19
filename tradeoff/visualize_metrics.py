import numpy as np
import pickle
import matplotlib.pyplot as plt



filename = "data/metrics4.pkl"
metrics = pickle.load(open(filename, "rb"))
metrics = np.asarray(metrics)

print(metrics)

feat_error = metrics[:,0]
unsure_perc = metrics[:,1]
sure_perc = metrics[:,2]
reward_error = metrics[:,3]
regret = metrics[:,4]

plt.plot(feat_error,'o-')
plt.plot(unsure_perc,'d--')
plt.plot(sure_perc,'s--')
plt.show()

plt.plot(reward_error,'o-')
plt.plot(regret,'d--')
plt.show()
