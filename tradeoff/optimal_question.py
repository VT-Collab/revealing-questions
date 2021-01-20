import numpy as np
from scipy.stats import multivariate_normal, uniform
import pickle


# hyperparameters:
# these are the max and min feature counts from the question set
# when you generate questions, these values are printed out
feat_min = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
feat_max = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
feat_min = np.asarray(feat_min)
feat_max = np.asarray(feat_max)



# generate a randomly sampled unit vector in 3D
def unit_vector():
    angle1 = np.random.uniform(0,np.pi*2)
    angle2 = np.arccos( np.random.uniform(-1,1) )
    x = np.sin( angle2) * np.cos( angle1 )
    y = np.sin( angle2) * np.sin( angle1 )
    z = np.cos( angle2)
    return np.asarray([x, y, z])

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
        theta_prop = theta_curr + np.random.normal(0, noise, 3)
        theta_prop /= np.linalg.norm(theta_prop)
        current_prob, proposed_prob = 1.0, 1.0
        for idx in range(len(questions)):
            current_prob *= boltzmann(answers[idx], questions[idx], theta_curr)
            proposed_prob *= boltzmann(answers[idx], questions[idx], theta_prop)
        if np.random.random() < proposed_prob / current_prob:
            theta_curr = np.copy(theta_prop)

# sampling algoritm we use to update Phi
# https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
def metropolis_hastings_phi(questions, burnin, phi_length, phi_start, forgetting_factor=1.0, noise=0.05):
    phi_curr = np.copy(phi_start)
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
            question_number = len(questions) - 1 - idx
            current_prob *= forgetting_factor**question_number * Qmodel.pdf(phi_curr)
            proposed_prob *= forgetting_factor**question_number * Qmodel.pdf(phi_prop)
        if np.random.random() < proposed_prob / current_prob:
            phi_curr = np.copy(phi_prop)

# input trajectory, output feature vector
def features(xi):
    dist2table = abs(0.1 - xi[-1][2]) / 0.6
    dist2goal = np.sqrt((0.8 - xi[-1][0])**2 + (-0.2 - xi[-1][1])**2) / np.sqrt(0.7**2 + 0.6**2)
    dist2obs_midpoint = abs(0.1 - xi[1][2]) / 0.6
    dist2obs_final = np.sqrt((0.6 - xi[-1][0])**2 + (0.1 - xi[-1][1])**2 + (0.1 - xi[-1][2])**2)  / np.sqrt(0.5**2 + 0.5**2 + 0.6**2)
    dist2obs = 0.5 * dist2obs_midpoint + 0.5 * dist2obs_final
    feature_vector = np.asarray([dist2table, dist2goal, dist2obs])
    return feature_vector

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
def gaussian(mean, cov=np.diag([0.1]*6)):
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
    Qmodel, M = gaussian(Q[-1]), len(Phi)
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
        score = Qinfo + Lambda * Qbelief
        if score > score_max:
            score_max = score
            Qopt = np.copy(Q)
    return Qopt

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

# metrics for the teaching aspects
def teaching_metrics(questionset, Theta, Phi):
    R_phi = theta2phi(questionset, Theta)
    H_phi = np.mean(Phi, axis=0)
    error = R_phi[0:3] - H_phi[0:3]             # H knows the expected feature counts
    total_sure, total_unsure = 0, 0
    idx_R_unsure = np.argmax(R_phi[3:6])        # H knows the feature R is most unsure about
    idx_R_sure = np.argmin(R_phi[3:6])          # H knows the feature R is most sure about
    for phi in Phi:
        if np.argmax(phi[3:6]) == idx_R_unsure:
            total_unsure += 1.0
        if np.argmin(phi[3:6]) == idx_R_sure:
            total_sure += 1.0
    return [np.linalg.norm(error), total_unsure / len(Phi), total_sure / len(Phi)]

# metrics for the learning aspects
def learning_metrics(questionset, Theta, theta_star):
    theta_error = theta_star - np.mean(Theta, axis=0)
    ideal_features = theta2phi(questionset, [theta_star])[0:3]
    actual_features = theta2phi(questionset, Theta)[0:3]
    regret = np.dot(theta_star, actual_features) - np.dot(theta_star, ideal_features)
    return [np.linalg.norm(theta_error), regret]




def main():

    # import the possible questions we have saved
    filename = "data/questions.pkl"
    questionset = pickle.load(open(filename, "rb"))

    # here are a couple hyperparameters we choose:
    n_questions = 20
    n_samples = 100
    burnin = 500
    Lambda = 2
    forgetting_factor = 0.75
    savename = "teaching"
    results = []

    for iteration in range(50):

        # here is what the human really thinks:
        theta_star = unit_vector()

        # at the start, the robot has a uniform prior over the human's reward
        Theta = uniform_prior_theta(n_samples)
        Phi = uniform_prior_phi(n_samples)
        questions = []
        answers = []
        metrics = []

        # at the start, the robot knows nothing:
        phi_star = theta2phi(questionset, Theta)

        # main loop --- here is where we find the questions
        for idx in range(n_questions):

            # get best question
            Qstar = optimal_question(questionset, Theta, Phi, phi_star, Lambda)

            # ask this question to the human, get their response
            p_IDK = boltzmann("idk", Qstar, theta_star)         # likelihood they think both are about the same
            p_A = boltzmann(Qstar[0], Qstar, theta_star)        # likelihood they pick the first option
            p_B = boltzmann(Qstar[1], Qstar, theta_star)        # likelihood they pick the second option
            q = np.random.choice(["idk", Qstar[0], Qstar[1]], p=[p_IDK, p_A, p_B])

            # update our list of questions and answers
            questions.append(Qstar)
            answers.append(q)
            pickle.dump(questions, open("data/optimal_questions.pkl", "wb"))
            pickle.dump(answers, open("data/human_answers.pkl", "wb"))

            # use metropolis hastings algorithm to update Phi
            Phi = metropolis_hastings_phi(questions, burnin, n_samples, phi_star, forgetting_factor)

            # metrics recording teaching
            metric_teaching = teaching_metrics(questionset, Theta, Phi)

            # use metropolis hastings algorithm to update Theta
            Theta = metropolis_hastings_theta(questions, answers, burnin, n_samples, theta_star)

            # metrics recording learning
            metric_learning = learning_metrics(questionset, Theta, theta_star)

            # update phi_star based on what the robot actually knows! (ALL method)
            phi_star = theta2phi(questionset, Theta)
            metrics.append(metric_teaching + metric_learning)

            # print off an update that we can read to check the progress
            print("[*] teaching metrics: ", metric_teaching)
            print("[*] learning metrics: ", metric_learning)
            print("[*] The human really wants: ", theta_star)
            print("[*] I think that theta* is: ", np.mean(Theta, axis=0))

        results.append(metrics)
        pickle.dump(results, open("data/" + savename + ".pkl", "wb"))
        print("[***] I just finised iteration: ", iteration)



if __name__ == "__main__":
    main()




# some code for most / one
        # most_uncertain = np.argmax(phi_star[3:6]) + 3
        # phi_most = np.asarray([phi_star[0], phi_star[1], phi_star[2], 0, 0, 0])
        # phi_most[most_uncertain] = phi_star[most_uncertain]
        # phi_star = np.copy(phi_most)
