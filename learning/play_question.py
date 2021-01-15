from env import SimpleEnv
import numpy as np
import time
import pickle
from scipy.interpolate import interp1d
import sys


class Trajectory(object):

    def __init__(self, xi, T):
        """ create interpolators between waypoints """
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



def main():

    questions = pickle.load(open("data/optimal_questions.pkl", "rb"))
    answers = pickle.load(open("data/human_answers.pkl", "rb"))
    for idx in range(len(questions)):
        if np.linalg.norm(answers[idx]- questions[idx][0]) < 1e-3:
            print("[*] the user answered with the one on the RIGHT\n")
        if np.linalg.norm(answers[idx]- questions[idx][1]) < 1e-3:
            print("[*] the user answered with the one on the LEFT\n")
        play_question(questions[idx])
        print("\n\n\n\n\n")



if __name__ == "__main__":
    main()
