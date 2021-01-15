import numpy as np
from scipy.stats import multivariate_normal, uniform
import pickle


# hyperparameters:
# these are the max and min feature counts from the question set
# when you generate questions, these values are printed out
feat_min = [0, 0, 0.4, 0, 0, 0]
feat_max = [0.6, 0.8, 1.2, 0.3, 0.4, 0.4]
feat_min = np.asarray(feat_min)
feat_max = np.asarray(feat_max)


# sampling algoritm we use to update Phi
# https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
def metropolis_hastings(questions, burnin, phi_length, noise=0.1):
    phi_curr = np.random.uniform(low=feat_min, high=feat_max)
    Phi = []
    while True:
        Phi.append(phi_curr)
        if len(Phi) == burnin + phi_length:
            Phi = np.asarray(Phi)
            return Phi[-phi_length:]
        phi_prop = np.copy(phi_curr) + np.random.uniform(low=-feat_max*noise, high=feat_max*noise)
        for idx in range(len(phi_prop)):
            if phi_prop[idx] > feat_max[idx]:
                phi_prop[idx] = feat_max[idx]
            if phi_prop[idx] < feat_min[idx]:
                phi_prop[idx] = feat_min[idx]
        current_prob, proposed_prob = 1.0, 1.0
        for idx in range(len(questions)):
            Qmodel = gaussian(questions[idx][-1])
            current_prob *= Qmodel.pdf(phi_curr)
            proposed_prob *= Qmodel.pdf(phi_prop)
        if np.random.random() < proposed_prob / current_prob:
            phi_curr = np.copy(phi_prop)

# likelihood (from the human's perspective) of robot choosing question Q
# given that the robot is thinking phi
# the variance Sigma is a hyperparameter we can play with
def gaussian(mean, cov=np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])):
    return multivariate_normal(mean, cov)

# (human's) uniform prior over what the robot knows
# values of phi are constrained to be between the max and min features from the questionset
def uniform_prior(M):
    Phi = []
    for idx in range(M):
        phi = np.random.uniform(low=feat_min, high=feat_max)
        Phi.append(phi)
    return np.asarray(Phi)

# compute the human's updated belief over what the robot knows
# this is the equation we derive in the paper
def belief(Q, Phi, phi_star):
    Qmodel, M = gaussian(Q[-1]), len(Phi)
    Z = 0
    for phi in Phi:
        Z += Qmodel.pdf(phi)
    Qbelief = Qmodel.pdf(phi_star) * M / Z
    return Qbelief

# identify the question that maximizes human's belief in what robot knows
def optimal_question(questionset, Phi, phi_star):
    Qopt, belief_max, count = None, 0.0, 0
    for Q in questionset:
        count += 1
        perc_complete = count * 100.0 / len(questionset)
        if not perc_complete % 10.0:
            print("[*] Percentage complete: ", perc_complete)
        Qbelief = belief(Q, Phi, phi_star)
        if Qbelief > belief_max:
            belief_max = Qbelief
            Qopt = np.copy(Q)
    return Qopt



def main():

    # here is what the robot really knows:
    phi_star = [0.4, 0.0, 1.0, 0.4, 0.0, 0.0]
    phi_star = np.asarray(phi_star)

    # here are a couple hyperparameters we choose:
    n_questions = 50
    n_samples = 100
    burnin = 500

    # import the possible questions we have saved
    filename = "data/questions.pkl"
    questionset = pickle.load(open(filename, "rb"))

    # at the start, the robot has a uniform prior over the human's reward
    Phi = uniform_prior(n_samples)
    questions = []

    # main loop --- here is where we find the questions
    for idx in range(n_questions):

        # get best question
        Qstar = optimal_question(questionset, Phi, phi_star)

        # update our list of questions and answers
        questions.append(Qstar)
        pickle.dump(questions, open("data/optimal_questions.pkl", "wb"))

        # use metropolis hastings algorithm to update Phi
        Phi = metropolis_hastings(questions, burnin, n_samples)

        # get the human's estimate of what phi_star is
        mean_phi = np.mean(Phi, axis=0)
        std_phi = np.std(Phi, axis=0)
        print("[*] The robot really thinks: ", phi_star)
        print("[*] The human thinks the robot thinks: ", mean_phi)
        print("[*] Here is what the human is confident about: ", std_phi)




if __name__ == "__main__":
    main()
