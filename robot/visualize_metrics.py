import numpy as np
import pickle
import matplotlib.pyplot as plt



def get_metric(data, index):
    metric = np.zeros((20, len(data)))
    for iteration in range(len(data)):
        X = np.asarray(data[iteration])
        metric[:, iteration] = X[:, index]
    mean_metric = np.mean(metric, axis=1)
    std_metric = np.std(metric, axis=1) / np.sqrt(len(data))
    return mean_metric, std_metric, metric

def get_all_metrics(data):
    mean_feat_error, std_feat_error, _ = get_metric(data, 0)
    # mean_std_error, std_std_error, _ = get_metric(data, 1)
    # mean_unsure_error, std_unsure_error, _ = get_metric(data, 2)
    # mean_sure_error, std_sure_error, _ = get_metric(data, 3)
    # mean_reward_error, std_reward_error, _ = get_metric(data, 10)
    mean_regret, std_regret, _ = get_metric(data, 11)
    mean_idk, std_idk, _ = get_metric(data, 12)
    return mean_feat_error, std_feat_error, mean_regret, std_regret, mean_idk, std_idk


filename = "data/random.pkl"
data = pickle.load(open(filename, "rb"))
Rfm, Rfs, Rrm, Rrs, Rim, Ris = get_all_metrics(data)

filename = "data/learning.pkl"
data = pickle.load(open(filename, "rb"))
Lfm, Lfs, Lrm, Lrs, Lim, Lis = get_all_metrics(data)

filename = "data/teaching.pkl"
data = pickle.load(open(filename, "rb"))
Tfm, Tfs, Trm, Trs, Tim, Tis = get_all_metrics(data)

filename = "data/tradeoff2.pkl"
data = pickle.load(open(filename, "rb"))
Ofm, Ofs, Orm, Ors, Oim, Ois = get_all_metrics(data)


plt.fill_between(range(20), Rfm-Rfs, Rfm+Rfs)
plt.fill_between(range(20), Lfm-Lfs, Lfm+Lfs)
plt.fill_between(range(20), Tfm-Tfs, Tfm+Tfs)
plt.fill_between(range(20), Ofm-Ofs, Ofm+Ofs)
# plt.title("feature error")
plt.show()

plt.fill_between(range(20), Rrm-Rrs, Rrm+Rrs)
plt.fill_between(range(20), Lrm-Lrs, Lrm+Lrs)
plt.fill_between(range(20), Trm-Trs, Trm+Trs)
plt.fill_between(range(20), Orm-Ors, Orm+Ors)
# plt.title("feature error")
plt.show()

plt.fill_between(range(20), Rim-Ris, Rim+Ris)
plt.fill_between(range(20), Lim-Lis, Lim+Lis)
plt.fill_between(range(20), Tim-Tis, Tim+Tis)
plt.fill_between(range(20), Oim-Ois, Oim+Ois)
# plt.title("feature error")
plt.show()

# small image of setting, plot of regret vs questions, plot of idks just for ours and info gain
