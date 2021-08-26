import numpy as np
from scipy.stats import multivariate_normal, uniform
import pickle
import rospy
import sys
import time
import playback_demo as pld
from gripper_control import GripperClient

# hyperparameters:
# these are the max and min feature counts from the question set
# when you generate questions, these values are printed out
feat_min = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
feat_max = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
feat_min = np.asarray(feat_min)
feat_max = np.asarray(feat_max)

# generate a randomly sampled unit vector in 2D
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
        theta_prop = theta_curr + np.random.normal(0, noise, len(theta_start))
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
def Q2features(Q, n_questions=2, n_features=3):
    F = np.zeros((n_questions, n_features))
    for idx in range(n_questions):
        F[idx,:] = features(Q[idx])
    features_mean = np.mean(F, axis=0)
    features_std = np.std(F, axis=0)
    return np.concatenate((features_mean, features_std))

# input trajectory, output feature vector
def features(xi):
    target = xi[-3]
    marker = xi[-2]
    distance_to_target = xi[-1]
    return np.asarray([target, marker, distance_to_target])

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
    elif np.linalg.norm(features(q) - features(Q[0])) < 1e-5:
        return 1/(1+np.exp(delta - beta * C(Q[1], theta) + beta * C(Q[0], theta)))
    elif np.linalg.norm(features(q) - features(Q[1])) < 1e-5:
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

# given what the robot has learned, predict the best trajectory for the human
def optimal_traj(questionset, Theta):
    theta_mean = np.mean(Theta, axis=0)
    xi_star, min_score = None, np.Inf
    for Q in questionset:
        for xi in [Q[0], Q[1]]:
            score = C(xi, theta_mean)
            if score < min_score:
                min_score = score
                xi_star = np.copy(xi)
    return xi_star

def demo(Q, duration, TO_START_TIME, sp, ep):
        playback = pld.TrajectoryClient(duration)
        playback.load_trajectory(Q[sp:ep],TO_START_TIME)
        playback.send(0)

def gripper(flag):
    trig = grip.GripperClient()
    if flag == False:
        trig.open_gripper()
    elif flag == True:
        trig.close_gripper()

def main():

    # here are the hyperparameters we are varying
    ask_random_questions = False    # random questions (baseline)
    rospy.init_node("play_trajectory")
    grip = GripperClient()
    # command lines to choose experiment type
    user_n = sys.argv[1]
    method = sys.argv[2]
    savename = "../Data/Study/user" + str(user_n)+ "/marker" + "_" + str(method) + ".pkl"

    if method == "ig":
        Lambda = [1, 0]                 # info gain (learning)
    elif method == "tf":
        Lambda = [1, 1]                 # trade-off (version 1)

    # import the possible questions we have saved
    filename = "../Data/Questions/Q_marker.pkl"
    questionset = pickle.load(open(filename, "rb"), encoding='latin1')

    # here are a couple hyperparameters we leave fixed:
    bounded_memory = 3
    n_questions = 15
    n_samples = 100
    burnin = 500
    results = []

    # here is what the human really thinks:
    theta_star = np.asarray([-1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])

    # at the start, the robot has a uniform prior over the human's reward
    Theta = uniform_prior_theta(n_samples)
    Phi = uniform_prior_phi(n_samples)
    questions = []
    answers = []
    user_data = np.zeros((n_questions,7))

    # at the start, the robot knows nothing:
    phi_star = theta2phi(questionset, Theta)

    # main loop --- here is where we find the questions
    q_count = 1
    for idx in range(n_questions):
        print("\n", "------The robot asks you question: ", q_count + idx)
        if ask_random_questions is True:
            # get random question
            Qstar = random_question(questionset)
        else:
            # get best question
            Qstar = optimal_question(questionset, Theta, Phi, phi_star, Lambda)

        """ here is where we play the question and get the human's answer """
        # duration of each trajectory
        duration_p = 3.5

        for i in range(2):
            if i == 0:
                TO_START_TIME = 3
            else:
                TO_START_TIME = 1
            print("Trajectory " + str(i+1))
            demo(Qstar[i], duration_p, TO_START_TIME, 0, 3)

        # ask this question to the human, get their response
        print("Feature 1 ----",Qstar[0][-3:], "----")
        print("Feature 2 ----",Qstar[1][-3:], "----")
        print("\n")

        """ here is where we play the question and get the human's answer """
        validinputs = ['1', '2', '3']
        userinput  = None
        while userinput not in validinputs:
            userinput = input("Please select the trajectory: ")
        if userinput == '1':
            q = Qstar[0]
        elif userinput == '2':
            q = Qstar[1]
        elif userinput == '3':
            q = "idk"
        """ here is where we play the question and get the human's answer """

        """ simulated human """
        # p_IDK = boltzmann("idk", Qstar, theta_star)         # likelihood they think both are about the same
        # p_A = boltzmann(Qstar[0], Qstar, theta_star)        # likelihood they pick the first option
        # p_B = boltzmann(Qstar[1], Qstar, theta_star)        # likelihood they pick the second option
        # q = np.random.choice(["idk", Qstar[0], Qstar[1]], p=[p_IDK, p_A, p_B])
        # print("boltzmann A: >>>",p_A,'\n',
        #     "boltzmann B: >>>", p_B, '\n',
        #     "boltzmann IDK: >>>", p_IDK,)
        """ simulated human """


        """ OK now back to the algoritm """
        # update our list of questions and answers
        questions.append(Qstar)
        answers.append(q)

        # use metropolis hastings algorithm to update Phi
        Phi = metropolis_hastings_phi(questions, burnin, n_samples, phi_star, bounded_memory)

        # use metropolis hastings algorithm to update Theta
        Theta = metropolis_hastings_theta(questions, answers, burnin, n_samples, theta_star)
        Theta[:,2] = np.random.uniform(-1, 1, n_samples)
        for j in range(n_samples):
            theta_magnitude = np.linalg.norm(Theta[j,:])
            Theta[j, :] /= theta_magnitude
        print("Robot's estimate of ThetaStar >>>>", np.mean(Theta, axis=0),'\n')

        # update phi_star based on what the robot actually knows! (ALL method)
        phi_star = theta2phi(questionset, Theta)

        # save the feature selections of the robot
        output = np.append([int(userinput)],
        np.concatenate((Qstar[0][-3:].tolist(), Qstar[1][-3:].tolist())))
        user_data[idx,:] = output
        print(int(userinput))

    # comnd = input("Press Enter to continue")
    # print("----->Robot shows user's preference<-----")
    # best_traj = optimal_traj(questionset, Theta)
    # demo(best_traj, 4, 1, 0, 3)

    pickle.dump(user_data, open(savename, "wb"))

if __name__ == "__main__":
    main()
