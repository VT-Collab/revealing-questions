from env import SimpleEnv
import numpy as np
import time
import pickle
from scipy.interpolate import interp1d
import sys


class Trajectory(object):

    def __init__(self, xi, T):
        """ create cublic interpolators between waypoints """
        self.xi = np.asarray(xi)
        self.T = T
        kind = "linear"
        self.n_waypoints = self.xi.shape[0]
        timesteps = np.linspace(0, self.T, self.n_waypoints)
        self.f1 = interp1d(timesteps, self.xi[:,0], kind=kind)
        self.f2 = interp1d(timesteps, self.xi[:,1], kind=kind)
        self.f3 = interp1d(timesteps, self.xi[:,2], kind=kind)

    def get(self, t):
        """ get interpolated position """
        if t < 0:
            q = [self.f1(0), self.f2(0), self.f3(0)]
        elif t < self.T:
            q = [self.f1(t), self.f2(t), self.f3(t)]
        else:
            q = [self.f1(self.T), self.f2(self.T), self.f3(self.T)]
        return np.asarray(q)



# play a trajectory over T seconds
def play_question(Q, T=3.0):
    traj1 = Trajectory(Q[0], T)
    traj2 = Trajectory(Q[1], T)
    env = SimpleEnv()
    state = env.reset()
    input("Press Enter to continue...")
    start_time = time.time()
    curr_time = time.time() - start_time
    while curr_time < T + 2.0:
        pos_desired1 = traj1.get(curr_time)
        pos_desired2 = traj2.get(curr_time)
        pos_desired = pos_desired1.tolist() + pos_desired2.tolist()
        next_state, reward, done, info = env.step(pos_desired)
        if done:
            break
        curr_time = time.time() - start_time
    env.close()



# pick what you want to replay:
# choices are all, one-turn, and one-most
# will show all trajectories in a row, were each pair of trajectories is a question
def main():

    type = sys.argv[1]
    number = sys.argv[2]
    filename = "data/optimal_questions-" + type + "-number" + number + ".pkl"
    Q_sequence = pickle.load(open(filename, "rb"))
    for Q in Q_sequence:
        play_question(Q)



if __name__ == "__main__":
    main()
