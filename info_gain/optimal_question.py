import numpy as np
import pickle


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
def metropolis_hastings(questions, answers, burnin, theta_length, noise=0.1):
    theta_curr = unit_vector()
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

# input trajectory, output feature vector
def features(xi):
    dist2table = abs(0.1 - xi[-1][2])
    dist2goal = np.sqrt((0.8 - xi[-1][0])**2 + (-0.2 - xi[-1][1])**2)
    dist2obs = abs(0.1 - xi[1][2])
    return np.asarray([dist2table, dist2goal, dist2obs])

# input trajectory and weights, output reward
def R(xi, theta):
    f = features(xi)
    return np.dot(theta, f)

# likelihood of human choosing answer q to question Q given reward weights theta
def boltzmann(q, Q, theta, beta=50.0):
    Z = 0
    for xi in Q:
        Z += np.exp(beta * R(xi, theta))
    return np.exp(beta * R(q, theta)) / Z

# uniform prior over the reward weights theta
# values of theta are constrained to be unit vectors
def uniform_prior(M):
    Theta = []
    for idx in range(M):
        thetap = unit_vector()
        Theta.append(thetap)
    return np.asarray(Theta)

# compute the info gain for a question using Equation (12)
def info_gain(Q, Theta):
    Qinfo, M = 0, len(Theta)
    for q in Q:
        Z = 0
        for theta in Theta:
            Z += boltzmann(q, Q, theta)
        for theta in Theta:
            Hmodel = boltzmann(q, Q, theta)
            Qinfo += 1/M * Hmodel * np.log2(M * Hmodel / Z)
    return Qinfo

# identify the question that maximizes information gain
def optimal_question(questionset, Theta):
    Qopt, info_max, count = None, 0.0, 0
    for Q in questionset:
        count += 1
        perc_complete = count * 100.0 / len(questionset)
        if not perc_complete % 10.0:
            print("[*] Percentage complete: ", perc_complete)
        Qinfo = info_gain(Q, Theta)
        if Qinfo > info_max:
            info_max = Qinfo
            Qopt = np.copy(Q)
    return Qopt



def main():

    # here is what the human really thinks:
    theta_star = [1, 0, 0]
    theta_star = np.asarray(theta_star)

    # here are a couple hyperparameters we choose:
    n_questions = 50
    n_samples = 100
    burnin = 500

    # import the possible questions we have saved
    filename = "data/questions.pkl"
    questionset = pickle.load(open(filename, "rb"))

    # at the start, the robot has a uniform prior over the human's reward
    Theta = uniform_prior(n_samples)
    questions = []
    answers = []

    # main loop --- here is where we find the questions
    for idx in range(n_questions):

        # get best question
        Q = optimal_question(questionset, Theta)

        # ask this question to the human, get their response
        p_A = boltzmann(Q[0], Q, theta_star)      # likelihood they pick the first option
        p_B = 1 - p_A                             # likelihood they pick the second option
        q = Q[ np.random.choice([0,1], p=[p_A, p_B]) ]

        # update our list of questions and answers
        questions.append(Q)
        answers.append(q)
        pickle.dump(questions, open("data/info_gain-questions.pkl", "wb"))
        pickle.dump(answers, open("data/info_gain-answers.pkl", "wb"))

        # use metropolis hastings algorithm to update Theta
        Theta = metropolis_hastings(questions, answers, burnin, n_samples)

        # get the robot's estimate of what theta_star is
        mean_theta = np.mean(Theta, axis=0)
        std_theta = np.std(Theta, axis=0)
        print("[*] The human really wants: ", theta_star)
        print("[*] I think the human wants: ", mean_theta)
        print("[*] Here is what I am confident about: ", std_theta)




if __name__ == "__main__":
    main()
