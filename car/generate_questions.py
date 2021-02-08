import numpy as np
import pickle



# input trajectory, output feature vector
def features(xi):
    distance = xi[-1][0]
    lane, avoid = 0, 1.0
    for idx in range(len(xi)):
        waypoint = xi[idx]
        lane += abs(waypoint[1])
        avoid = min(avoid, np.sqrt((2 - waypoint[0])**2 + (-0.5 - waypoint[1])**2))
    feature_vector = np.asarray([(distance - 1.5) / 2.5, lane / 4.0, avoid])
    return feature_vector

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
    n_waypoints = 2
    n_questions = 5e2
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
        Q.append(Qfeatures(Q))
        dataset.append(Q)

    pickle.dump(dataset, open(savename, "wb"))
    print("[*] I just saved this many questions: ", len(dataset))

    F = []
    for Q in dataset:
        F.append(Q[-1])
    F = np.asarray(F)
    print("mean features: ", np.mean(F, axis=0))
    print("stdv features: ", np.std(F, axis=0))

    mins, maxs = [] ,[]
    for idx in range(6):
        mins.append(np.min(F[:,idx]))
        maxs.append(np.max(F[:,idx]))
    print("min features: ", np.asarray(mins))
    print("max features: ", np.asarray(maxs))

if __name__ == "__main__":
    main()
