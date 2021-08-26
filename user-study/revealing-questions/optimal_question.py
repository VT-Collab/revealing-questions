import numpy as np
from scipy.stats import multivariate_normal, uniform
import pickle


# hyperparameters:
# these are the max and min feature counts from the question set
# when you generate questions, these values are printed out
feat_min = [0.0, 0.0, 0.0, 0.0]
feat_max = [1.0, 1.0, 0.5, 0.5]
feat_min = np.asarray(feat_min)
feat_max = np.asarray(feat_max)



# generate a randomly sampled unit vector in 2D
def unit_vector():
    angle = np.random.uniform(0,np.pi*2)
    x = np.cos( angle )
    y = np.sin( angle )
    return np.asarray([x, y])

# sampling algoritm we use to update Theta
# https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
def metropolis_hastings_theta(questions, answers, burnin, theta_length, theta_start, noise=0.05):
    theta_curr = np.copy(theta_start)
    Theta = []
    while True:
        Theta.append(theta_curr)
        if len(Theta) == burnin + theta_length:
            Theta = np.asarray(Theta)
            return Theta[-theta_length:]
        theta_prop = theta_curr + np.random.normal(0, noise, len(theta_star))
        theta_prop /= np.linalg.norm(theta_prop)
        current_prob, proposed_prob = 1.0, 1.0
        for idx in range(len(questions)):
            current_prob *= boltzmann(answers[idx], questions[idx], theta_curr)
            proposed_prob *= boltzmann(answers[idx], questions[idx], theta_prop)
        if np.random.random() < proposed_prob / current_prob:
            theta_curr = np.copy(theta_prop)

# sampling algoritm we use to update Phi
# https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
def metropolis_hastings_phi(questions, burnin, phi_length, phi_start, bounded_memory=3, noise=0.05):
    phi_curr = np.copy(phi_start)
    Phi = []
    last_question = max([0, len(questions) - bounded_memory])
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
        for idx in range(last_question, len(questions)):
            Q_features = Q2features(questions[idx])
            Qmodel = gaussian(Q_features)
            current_prob *= Qmodel.pdf(phi_curr)
            proposed_prob *= Qmodel.pdf(phi_prop)
        if np.random.random() < proposed_prob / current_prob:
            phi_curr = np.copy(phi_prop)

# input question, output mean and variance over features
def Q2features(Q, n_questions=2, n_features=2):
    F = np.zeros((n_questions, n_features))
    for idx in range(n_questions):
        F[idx,:] = features(Q[idx])
    features_mean = np.mean(F, axis=0)
    features_std = np.std(F, axis=0)
    return np.concatenate((features_mean, features_std))

# input trajectory, output feature vector
def features(xi):
    height = xi[-2]
    distance_to_target = xi[-1]
    return np.asarray([height, distance_to_target])

# input trajectory and weights, output cost
def C(xi, theta):
    f = features(xi)
    return np.dot(theta, f)

# likelihood of human choosing answer q to question Q given reward weights theta
def boltzmann(q, Q, theta, beta=50.0, delta=1.0):
    if q is "idk":
        pq1 = 1/(1+np.exp(delta - beta * C(Q[1], theta) + beta * C(Q[0], theta)))
        pq2 = 1/(1+np.exp(delta - beta * C(Q[0], theta) + beta * C(Q[1], theta)))
        return (np.exp(2*delta)-1)*pq1*pq2
    elif q is Q[0]:
        return 1/(1+np.exp(delta - beta * C(Q[1], theta) + beta * C(Q[0], theta)))
    elif q is Q[1]:
        return 1/(1+np.exp(delta - beta * C(Q[0], theta) + beta * C(Q[1], theta)))

# likelihood (from the human's perspective) of robot choosing question Q
# given that the robot is thinking phi
# the variance Sigma is a hyperparameter we can play with
def gaussian(mean, cov=np.diag([0.1]*len(feat_min))):
    return multivariate_normal(mean, cov)

# uniform prior over the reward weights theta
# values of theta are constrained to be unit vectors
def uniform_prior_theta(M):
    Theta = []
    for idx in range(M):
        theta = unit_vector()
        Theta.append(theta)
    return np.asarray(Theta)

# (human's) uniform prior over what the robot knows
# values of phi are constrained to be between the max and min features from the questionset
def uniform_prior_phi(M):
    Phi = []
    for idx in range(M):
        phi = np.random.uniform(low=feat_min, high=feat_max)
        Phi.append(phi)
    return np.asarray(Phi)

# compute the info gain for a question using Equation (12)
def info_gain(Q, Theta):
    Qinfo, M = 0, len(Theta)
    for q in ["idk", Q[0], Q[1]]:
        Z = 0
        for theta in Theta:
            Z += boltzmann(q, Q, theta)
        for theta in Theta:
            Hmodel = boltzmann(q, Q, theta)
            Qinfo += 1/M * Hmodel * np.log2(M * Hmodel / Z)
    return Qinfo

# compute the human's updated belief over what the robot knows
# this is the equation we derive in the paper
def belief(Q, Phi, phi_star):
    Q_features = Q2features(Q)
    Qmodel, M = gaussian(Q_features), len(Phi)
    Z = 0
    for phi in Phi:
        Z += Qmodel.pdf(phi)
    Qbelief = Qmodel.pdf(phi_star) / (Z / M + Qmodel.pdf(phi_star))
    return Qbelief

# identify the question that maximizes information gain AND
# maximizes human's belief in what robot knows
def optimal_question(questionset, Theta, Phi, phi_star, Lambda):
    Qopt, score_max, count = None, 0.0, 0
    for Q in questionset:
        count += 1
        perc_complete = count * 100.0 / len(questionset)
        if not perc_complete % 10.0:
            print("[*] Percentage complete: ", perc_complete)
        Qinfo = info_gain(Q, Theta)
        Qbelief = belief(Q, Phi, phi_star)
        score = Lambda[0] * Qinfo + Lambda[1] * Qbelief
        if score > score_max:
            score_max = score
            Qopt = np.copy(Q)
    return Qopt

# default scheme with random questions
def random_question(questionset):
    idx = np.random.choice(range(len(questionset)))
    return np.copy(questionset[idx])

# given the samples Theta, parameterize the robot's belief as phi
def theta2phi(questionset, Theta):
    F = []
    for theta in Theta:
        xi_star, min_score = None, np.Inf
        for Q in questionset:
            for xi in [Q[0], Q[1]]:
                score = C(xi, theta)
                if score < min_score:
                    min_score = score
                    xi_star = np.copy(xi)
        F.append(features(xi_star))
    features_mean = np.mean(F, axis=0)
    features_std = np.std(F, axis=0)
    return np.concatenate((features_mean, features_std))


def main():

    # here are the hyperparameters we are varying
    savename = "random"
    ask_random_questions = True    # random questions (baseline)
    Lambda = [1, 0]                 # info gain (learning)
    Lambda = [0, 1]                 # belief_h (teaching)
    Lambda = [1, 1]                 # trade-off (version 1)
    Lambda = [1, 2]                 # trade-off (version 2)

    # import the possible questions we have saved
    filename = "Data/Q_plate_20.pkl"
    questionset = pickle.load(open(filename, "rb"))

    # here are a couple hyperparameters we leave fixed:
    bounded_memory = 3
    n_questions = 20
    n_samples = 100
    burnin = 500
    results = []

    # here is what the human really thinks:
    theta_star = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)])

    # at the start, the robot has a uniform prior over the human's reward
    Theta = uniform_prior_theta(n_samples)
    Phi = uniform_prior_phi(n_samples)
    questions = []
    answers = []

    # at the start, the robot knows nothing:
    phi_star = theta2phi(questionset, Theta)

    # main loop --- here is where we find the questions
    for idx in range(n_questions):

        if ask_random_questions is True:
            # get random question
            Qstar = random_question(questionset)
        else:
            # get best question
            Qstar = optimal_question(questionset, Theta, Phi, phi_star, Lambda)


        """ here is where we play the question and get the human's answer """

        # ask this question to the human, get their response
        p_IDK = boltzmann("idk", Qstar, theta_star)         # likelihood they think both are about the same
        p_A = boltzmann(Qstar[0], Qstar, theta_star)        # likelihood they pick the first option
        p_B = boltzmann(Qstar[1], Qstar, theta_star)        # likelihood they pick the second option
        q = np.random.choice(["idk", Qstar[0], Qstar[1]], p=[p_IDK, p_A, p_B])


        """ OK now back to the algoritm """

        # update our list of questions and answers
        questions.append(Qstar)
        answers.append(q)

        # use metropolis hastings algorithm to update Phi
        Phi = metropolis_hastings_phi(questions, burnin, n_samples, phi_star, forgetting_factor)

        # use metropolis hastings algorithm to update Theta
        Theta = metropolis_hastings_theta(questions, answers, burnin, n_samples, theta_star)

        # update phi_star based on what the robot actually knows! (ALL method)
        phi_star = theta2phi(questionset, Theta)

    pickle.dump(results, open("data/" + savename + ".pkl", "wb"))



if __name__ == "__main__":
    main()
