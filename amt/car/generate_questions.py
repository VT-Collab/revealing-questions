import numpy as np
import pickle



# 3 features:
# distance traveled forward (want to maximize)
# distance from center of lane (want to minimize)
# xy distance from the obstacle (want to be above threshold)
# object locations are hard coded
def get_features(xi):
    distance = xi[-1][0]
    lane, avoid = 0, 1.0
    for idx in range(len(xi)):
        waypoint = xi[idx]
        lane += abs(waypoint[1])
        avoid = min(avoid, np.sqrt((2 - waypoint[0])**2 + (-0.4 - waypoint[1])**2))
    return np.asarray([distance, lane, avoid])



# get the mean and stdev of the feature counts in a question Q
# return a vector [mean, stdev]
def get_feature_vector(Q, n_features=3):
    n_questions = len(Q)
    F = np.zeros((n_questions, n_features))
    for idx in range(n_questions):
        F[idx,:] = get_features(Q[idx])
    Q_phi_mean = np.mean(F, axis=0)
    Q_phi_std = np.std(F, axis=0)
    Q_phi = np.concatenate((Q_phi_mean, Q_phi_std))
    return Q_phi.tolist()



def main():

    dataset = []
    savename = 'data/questions.pkl'
    n_waypoints = 2
    n_questions = 1e4
    n_choices = 2

    for question in range(int(n_questions)):
        Q = []
        for q in range(n_choices):
            xi = np.zeros((n_waypoints, 2))
            x_pos = 0.0
            for waypoint in range(n_waypoints):
                # sample position
                step = [np.random.uniform(low=0.75, high=2.0), np.random.uniform(low=-2.0, high=0.0)]
                x_pos += step[0]
                xi[waypoint,:] = [x_pos, step[1]]
                # impose workspace limits
                if xi[waypoint, 1] < -2:
                    xi[waypoint, 1] = -2
                if xi[waypoint, 1] > 2:
                    xi[waypoint, 1] = 2
            # add trajectory to question
            Q.append(xi.tolist())
        # get mean + stdev of the question features
        Q.append(get_feature_vector(Q))
        dataset.append(Q)

    pickle.dump(dataset, open(savename, "wb"))
    print("[*] I just saved this many questions: ", len(dataset))

    features = []
    for Q in dataset:
        features.append(Q[-1])
    features = np.asarray(features)
    print("mean features: ", np.mean(features, axis=0))
    print("stdv features: ", np.std(features, axis=0))

    mins, maxs = [] ,[]
    for idx in range(6):
        mins.append(np.min(features[:,idx]))
        maxs.append(np.max(features[:,idx]))
    print("min features: ", np.asarray(mins))
    print("max features: ", np.asarray(maxs))



if __name__ == "__main__":
    main()
