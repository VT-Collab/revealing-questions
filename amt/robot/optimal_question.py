import numpy as np
import pickle
from scipy.stats import multivariate_normal
import sys



# enter the range of phi ---
# this is used for normalization (since some features have different
# ranges than others)
# true_low=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
true_low=np.asarray([0., 0.01, 0., 0., 0., 0.])
# true_high=np.asarray([0.6, 0.8, 0.9, 0.3, 0.4, 0.4])
# true_high=np.asarray([0.6, 1.4, 1.5, 0.3, 0.5, 0.6])
true_high=np.asarray([0.6, 0.92, 0.6, 0.3, 0.46, 0.3])



# human's prior (i.e., what the human expects phi to be coming in)
bh_mean = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
bh_var = [0.02, 0.02, 0.02, 0.005, 0.005, 0.005]
prior = multivariate_normal(mean=bh_mean, cov=np.diag(bh_var))



# what the robot actually knows and does not know (phi)
# phi_true = np.asarray([0.3, 0.3, 0.4, 0.1, 0.2, 0.3])

# Phi for perfect demo
# phi_true = np.asarray([0., 0., 0.5, 0., 0., 0.])

# Phi with best questions
# phi_true = np.asarray([0.4, 0.4, 0.1, 0.4, 0.4, 0.4])

phi_true = np.asarray([0., 0., 0., 0.02, 0.02, 0.4])

# hyperparameters
Sigma = np.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
n_phi_samples = 50
n_iterations = 3
burnin= 100



# metropolis hastings algorithm
# https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
# this samples a set of phi the human thinks are likely given the
# questions asked so far
def metropolis_hastings(Q_set):
    current_phi = np.random.uniform(true_low, true_high)
    Omega = []
    while True:
        Omega.append(current_phi.tolist())
        # if we have enough samples, return the last n_phi_samples
        if len(Omega) == burnin + n_phi_samples:
            Omega = np.asarray(Omega)
            return Omega[-n_phi_samples:]
        # generate a new proposed value for phi
        movement = np.random.uniform(low=-true_high/10.0, high=true_high/10.0)
        proposed_phi = current_phi + movement
        # ensure proposed value in range of possible values
        for idx in range(6):
            if proposed_phi[idx] > true_high[idx]:
                proposed_phi[idx] = true_high[idx]
            if proposed_phi[idx] < true_low[idx]:
                proposed_phi[idx] = true_low[idx]
        # compute unnormalized likelihood of phi given prior and questions so far
        current_prob, proposed_prob = prior.pdf(current_phi), prior.pdf(proposed_phi)
        for Q in Q_set:
            distribution = multivariate_normal(mean=Q[-1], cov=Sigma)
            current_prob *= distribution.pdf(current_phi)
            proposed_prob *= distribution.pdf(proposed_phi)
        # use or reject the proposed phi
        acceptance = min(proposed_prob / current_prob, 1)
        if np.random.random() < acceptance:
            current_phi = np.copy(proposed_phi)



# solve for the question (in dataset) that maximizes the human's belief in the true phi
# given that Omega is a set of phi values the human currently thinks are likely
def get_question(dataset, Omega, phi):
    Q_max, p_max, count = None, 0, 0
    for Q in dataset:
        # compute the normalizer Z = sum P(Q | phi')
        Z, distribution = 0, multivariate_normal(mean=Q[-1], cov=Sigma)
        for phiprime in Omega:
            Z += distribution.pdf(phiprime)
        # get P(phi | Q) = P(Q | phi) / (1/N * sum P(Q | phi'))
        p = distribution.pdf(phi) * len(Omega) / Z
        # save the question Q that maximizes p
        if p > p_max:
            p_max = p
            Q_max = np.copy(Q)
        # print out completion percentage (since this part can be slow)
        count += 1
        perc_complete = count * 100.0 / len(dataset)
        if not perc_complete % 10.0:
            print("[*] Percentage complete: ", perc_complete)
    return Q_max



def main(method, i):

    # there the 3 ways to choose the questions:
    #   1. pick the question that conveys every aspect of phi at once (all)
    #   2. pick questions that convey aspects of phi one at a time (one-turn)
    #   3. pick questions that focus on the part of phi the human is most confused about (one-most)
    # type = sys.argv[1]
    # number = sys.argv[2]
    type = method
    number = str(i)
    filename = "data/questions.pkl"
    savename = "data/optimal_questions-" + type + "-number" + number + ".pkl"
    dataset = pickle.load(open(filename, "rb"))

    # sample from prior to make Omega
    Omega = np.zeros((n_phi_samples, len(phi_true)))
    for item in range(n_phi_samples):
        proposed_phi = np.random.multivariate_normal(bh_mean, np.diag(bh_var))
        for idx in range(len(phi_true)):
            if proposed_phi[idx] > true_high[idx]:
                proposed_phi[idx] = true_high[idx]
            if proposed_phi[idx] < true_low[idx]:
                proposed_phi[idx] = true_low[idx]
        Omega[item,:] = proposed_phi
    mean_human_belief = np.mean(Omega, axis=0)
    print("[*] human's estimate of phi: ", mean_human_belief)

    # depending on the question type, pick the phi we will try to teach to the human
    phi = np.copy(phi_true)
    if type == "one-turn":
        feature_turn = 0
        phi[3:6] = np.asarray([0, 0, 0])
        phi[3 + feature_turn] = phi_true[3 + feature_turn]
    elif type == "one-most":
        error = abs(phi_true - mean_human_belief)
        feature_most = np.argmax(error[0:3] + 10 * error[3:6])
        phi[3:6] = np.asarray([0, 0, 0])
        phi[3 + feature_most] = phi_true[3 + feature_most]

    # loop through iterations (Main loop)
    Q_sequence = []
    for iter in range(n_iterations):
        print("[*] I'm trying to teach: ", phi)
        Q = get_question(dataset, Omega, phi)
        print("[*] trajectory 1: ", Q[0])
        print("[*] trajectory 2: ", Q[1])
        Q_sequence.append(Q)
        pickle.dump(Q_sequence, open(savename, "wb"))
        Omega = metropolis_hastings(Q_sequence)
        mean_human_belief = np.mean(Omega, axis=0)
        print("[*] human's estimate of phi: ", mean_human_belief)

        # depending on the question type, pick the phi we will try to teach to the human
        phi = np.copy(phi_true)
        if type == "one-turn":
            feature_turn += 1
            feature_turn = feature_turn % 3
            phi[3:6] = np.asarray([0, 0, 0])
            phi[3 + feature_turn] = phi_true[3 + feature_turn]
        elif type == "one-most":
            error = abs(phi_true - mean_human_belief)
            feature_most = np.argmax(error[0:3] + 10 * error[3:6])
            phi[3:6] = np.asarray([0, 0, 0])
            phi[3 + feature_most] = phi_true[3 + feature_most]


if __name__ == "__main__":
    # methods = ["all", "one-most", "one-turn"]
    methods = ["one-turn"]
    for method in methods:
        for i in range(1,2):
            main(method, i)
