import numpy as np
import pickle
import matplotlib.pyplot as plt



def get_metric(data, index):
    metric = np.zeros((20, len(data)))
    for iteration in range(len(data)):
        X = np.asarray(data[iteration])
        metric[:, iteration] = X[:, index]
    mean_metric = np.mean(metric, axis=1)
    std_metric = np.std(metric, axis=1)
    return mean_metric, std_metric, metric

def get_all_metrics(data):
    mean_feat_error, std_feat_error, _ = get_metric(data, 0)
    mean_unsure_perc, std_unsure_perc, _ = get_metric(data, 1)
    mean_sure_perc, std_sure_perc, _ = get_metric(data, 2)
    mean_reward_error, std_reward_error, _ = get_metric(data, 3)
    mean_regret, std_regret, _ = get_metric(data, 4)
    return mean_feat_error, mean_unsure_perc, mean_sure_perc, mean_reward_error, mean_regret


filename = "data/learning.pkl"
data = pickle.load(open(filename, "rb"))
Lmean_feat_error, Lmean_unsure_perc, Lmean_sure_perc, Lmean_reward_error, Lmean_regret = get_all_metrics(data)

filename = "data/teaching.pkl"
data = pickle.load(open(filename, "rb"))
Tmean_feat_error, Tmean_unsure_perc, Tmean_sure_perc, Tmean_reward_error, Tmean_regret = get_all_metrics(data)


plt.plot(Lmean_feat_error,'o--')
plt.plot(Tmean_feat_error,'s-')
plt.show()

plt.plot(Lmean_unsure_perc,'o--')
plt.plot(Tmean_unsure_perc,'s-')
plt.show()

plt.plot(Lmean_sure_perc,'o--')
plt.plot(Tmean_sure_perc,'s-')
plt.show()

plt.plot(Lmean_reward_error,'o--')
plt.plot(Tmean_reward_error,'s-')
plt.show()

plt.plot(Lmean_regret,'o--')
plt.plot(Tmean_regret,'s-')
plt.show()
