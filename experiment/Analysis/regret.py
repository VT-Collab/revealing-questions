import numpy as np
import pickle
import optimal_question as opt
import sys
import optimal_question as resc
import matplotlib.pyplot as plt


filename = "../Data/Questions/Q_plate.pkl"
questionset = pickle.load(open(filename, "rb"), encoding='latin1')

""" compute the maximum reward """
theta_star = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)])
xi_star = opt.optimal_traj(questionset,theta_star)
f_star = opt.features(xi_star)
R_max = f_star[0]*theta_star[0] + f_star[1]*theta_star[1]
""" compute the maximum reward """

def compute_R(theta):
    xi_star = opt.optimal_traj(questionset,theta)
    f_star = opt.features(xi_star)
    theta_star = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)])
    return f_star[0]*theta_star[0] + f_star[1]*theta_star[1]


def compute_regret(Theta):
    regret = []
    for theta in Theta:
        R = compute_R(theta)
        cost = R - R_max
        regret.append(cost)
    return regret

def mean_regret(user):
    iter = 100
    regret_tf = []
    regret_ig = []
    for idx in range(iter):
        print("[*]iteration: ", str(idx))
        for method in ['tf', 'ig']:
            print("User: ", user)
            print("Method: ",method, '\n')
            Theta = resc.main(str(user), method)
            regret = compute_regret(Theta)
            if method == 'tf':
                regret_tf.append(regret)
            else:
                regret_ig.append(regret)
    regret_tf = np.mean(regret_tf, axis=0)
    regret_ig = np.mean(regret_ig, axis=0)
    return regret_tf, regret_ig


""" reproducing Theta of the user study """
all_tf = []
all_ig = []
for user in range(1,11):
    regret_tf, regret_ig = mean_regret(user)
    all_tf.append(regret_tf)
    all_ig.append(regret_ig)

    # print(type(all_tf))
    # print(regret_ig)
""" reproducing Theta of the user study """


""" This plots the regret for all users"""
fig, axs = plt.subplots(1, 2)

for user in all_ig:
    q_ig = list(range(1,len(user)+1))
    axs[0].plot(q_ig, user)
for user in all_tf:
    q_tf = list(range(1,len(user)+1))
    axs[1].plot(q_tf, user)

axs[0].set_title('Informative')
axs[1].set_title('Ours')
for ax in axs.flat:
    ax.set(xlabel='Number of Questions', ylabel='Regret')
plt.show()
""" This plots the regret for all users"""
