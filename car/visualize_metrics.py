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
    print(std_metric)
    return mean_metric, std_metric, metric

def get_all_metrics(data):
    mean_feat_error, std_feat_error, _ = get_metric(data, 0)
    mean_std_error, std_std_error, _ = get_metric(data, 1)
    mean_unsure_error, std_unsure_error, _ = get_metric(data, 2)
    mean_sure_error, std_sure_error, _ = get_metric(data, 3)
    mean_reward_error, std_reward_error, _ = get_metric(data, 10)
    mean_regret, std_regret, _ = get_metric(data, 11)
    mean_idk, std_idk, _ = get_metric(data, 12)
    return mean_feat_error, std_feat_error, mean_std_error, std_std_error, mean_unsure_error, mean_sure_error, mean_reward_error, mean_regret, mean_idk


filename = "data/random.pkl"
data = pickle.load(open(filename, "rb"))
Rmean_feat_error, Rstdfeat, Rmean_std_perc, Rstdstd, Runsure, Rsure, Rmean_reward_error, Rmean_regret, Ridk = get_all_metrics(data)

filename = "data/learning.pkl"
data = pickle.load(open(filename, "rb"))
Lmean_feat_error, Lstdfeat, Lmean_std_perc, Lstdstd, Lunsure,Lsure, Lmean_reward_error, Lmean_regret, Lidk = get_all_metrics(data)

filename = "data/teaching.pkl"
data = pickle.load(open(filename, "rb"))
Tmean_feat_error, Tstdfeat, Tmean_std_perc, Tstdstd, Tunsure, Tsure, Tmean_reward_error, Tmean_regret, Tidk = get_all_metrics(data)

filename = "data/tradeoff1.pkl"
data = pickle.load(open(filename, "rb"))
O1mean_feat_error, O1stdfeat, O1mean_std_perc, O1stdstd, O1unsure, O1sure, O1mean_reward_error, O1mean_regret, O1idk = get_all_metrics(data)

filename = "data/tradeoff2.pkl"
data = pickle.load(open(filename, "rb"))
O2mean_feat_error, O2stdfeat, O2mean_std_perc, O2stdstd, O2unsure, O2sure, O2mean_reward_error, O2mean_regret, O2idk = get_all_metrics(data)


plt.plot(Rmean_feat_error,'v--')
plt.fill_between(range(20), Rmean_feat_error-Rstdfeat, Rmean_feat_error+Rstdfeat)
plt.plot(Lmean_feat_error,'o--')
plt.fill_between(range(20), Lmean_feat_error-Lstdfeat, Lmean_feat_error+Lstdfeat)
plt.plot(Tmean_feat_error,'+-')
plt.fill_between(range(20), Tmean_feat_error-Tstdfeat, Tmean_feat_error+Tstdfeat)
plt.plot(O1mean_feat_error,'d-')
plt.fill_between(range(20), O1mean_feat_error-O1stdfeat, O1mean_feat_error+O1stdfeat)
plt.plot(O2mean_feat_error,'s-')
plt.fill_between(range(20), O2mean_feat_error-O2stdfeat, O2mean_feat_error+O2stdfeat)
plt.title("feature error")
plt.show()

plt.plot(Rmean_std_perc,'v--')
plt.fill_between(range(20), Rmean_std_perc-Rstdstd, Rmean_std_perc+Rstdstd)
plt.plot(Lmean_std_perc,'o--')
plt.fill_between(range(20), Lmean_std_perc-Lstdstd, Lmean_std_perc+Lstdstd)
plt.plot(Tmean_std_perc,'+-')
plt.fill_between(range(20), Tmean_std_perc-Tstdstd, Tmean_std_perc+Tstdstd)
plt.plot(O1mean_std_perc,'d-')
plt.fill_between(range(20), O1mean_std_perc-O1stdstd, O1mean_std_perc+O1stdstd)
plt.plot(O2mean_std_perc,'s-')
plt.fill_between(range(20), O2mean_std_perc-O2stdstd, O2mean_std_perc+O2stdstd)
plt.title("std error")
plt.show()

plt.plot(Rsure, Runsure, 'v')
plt.plot(Lsure, Lunsure, 'o')
plt.plot(Tsure, Tunsure, '+')
plt.plot(O1sure, O1unsure, 'd')
plt.plot(O2sure, O2unsure, 's')
plt.title("% know most confident vs. % know must unsure")
plt.show()

plt.plot(Rmean_reward_error,'v--')
plt.plot(Lmean_reward_error,'o--')
plt.plot(Tmean_reward_error,'+-')
plt.plot(O1mean_reward_error,'d-')
plt.plot(O2mean_reward_error,'s-')
plt.title("reward error")
plt.show()

plt.plot(Rmean_regret,'v--')
plt.plot(Lmean_regret,'o--')
plt.plot(Tmean_regret,'+-')
plt.plot(O1mean_regret,'d-')
plt.plot(O2mean_regret,'s-')
plt.title("regret")
plt.show()

plt.plot(Ridk,'v--')
plt.plot(Lidk,'o--')
plt.plot(Tidk,'+-')
plt.plot(O1idk,'d-')
plt.plot(O2idk,'s-')
plt.title("# of idks")
plt.show()
