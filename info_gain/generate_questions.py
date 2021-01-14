import numpy as np
import pickle



def main():

    dataset = []
    savename = 'data/questions.pkl'
    n_waypoints = 3
    n_questions = 1e3
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
        dataset.append(Q)

    pickle.dump(dataset, open(savename, "wb"))
    print("[*] I just saved this many questions: ", len(dataset))


if __name__ == "__main__":
    main()
