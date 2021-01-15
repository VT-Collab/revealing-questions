import numpy as np
import pickle


# input trajectory, output feature vector
def features(xi):
    dist2table = abs(0.1 - xi[-1][2])
    dist2goal = np.sqrt((0.8 - xi[-1][0])**2 + (-0.2 - xi[-1][1])**2)
    dist2obs = abs(0.1 - xi[1][2]) + np.sqrt((0.6 - xi[-1][0])**2 + (0.1 - xi[-1][1])**2 + (0.1 - xi[-1][2])**2)
    return np.asarray([dist2table, dist2goal, dist2obs])

# input question, output vector with feature mean and variance
def Qfeatures(Q, n_questions=2, n_features=3):
    F = np.zeros((n_questions, n_features))
    for idx in range(n_questions):
        F[idx,:] = features(Q[idx])
    features_mean = np.mean(F, axis=0)
    features_std = np.std(F, axis=0)
    return np.concatenate((features_mean, features_std))



def main():

    dataset = []
    savename = 'data/questions.pkl'
    n_waypoints = 3
    n_questions = 1e2
    n_choices = 2

    for question in range(int(n_questions)):
        Q = []
        for q in range(n_choices):
            xi = np.zeros((n_waypoints, 3))
            xi[0,:] = np.asarray([0.3, 0.9, 0.5])
            for waypoint in range(1, n_waypoints):
                if waypoint == 1:
                    h = np.random.normal(0.4,0.2)
                    step = [0.6, 0.1, h]
                else:
                    step = np.random.multivariate_normal([0.6, 0.1, 0.2], np.diag([0.3, 0.3, 0.2]))
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
        Q.append(Qfeatures(Q))
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
