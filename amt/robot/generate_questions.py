import numpy as np
import pickle



# 3 features:
# height from the table
# xy distance from the goal (a red bowl)
# xy distance from the obstacle (a cracker box)
# object locations are hard coded
def get_features(xi):
    height, dist2goal, dist2obs = 0, 0, 0
    for idx in range(1, len(xi)):
        waypoint = xi[idx]
        height += abs(0.1 - waypoint[2])
        dist2goal += np.sqrt((0.7 - waypoint[0])**2 + (-0.2 - waypoint[1])**2)
        dist2obs += np.sqrt((0.5 - waypoint[0])**2 + (0.2 - waypoint[1])**2 + (0.1 - waypoint[2])**2)
    return np.asarray([height, dist2goal, dist2obs])



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
            xi = np.zeros((n_waypoints, 3))
            # set start position
            xi[0,:] = np.asarray([0.5545, 0.0, 0.5195])
            for waypoint in range(1, n_waypoints):
                # sample next position (goal position)
                step = np.random.multivariate_normal([0.5, 0.0, 0.3], np.diag([0.1, 0.1, 0.1]))
                xi[waypoint,:] = step
                # impose workspace limits
                if xi[waypoint, 0] < 0.1:
                    xi[waypoint, 0] = 0.1
                if xi[waypoint, 0] > 0.8:
                    xi[waypoint, 0] = 0.8
                if xi[waypoint, 1] < -0.4:
                    xi[waypoint, 1] = -0.4
                if xi[waypoint, 1] > 0.4:
                    xi[waypoint, 1] = 0.4
                if xi[waypoint, 2] < 0.1:
                    xi[waypoint, 2] = 0.1
                if xi[waypoint, 2] > 0.7:
                    xi[waypoint, 2] = 0.7
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
