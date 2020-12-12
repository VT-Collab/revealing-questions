from env import SimpleEnv
import numpy as np
import time
import pickle
import sys


# play a trajectory over T seconds
def play_question(Q, speed=10.0):
    env = SimpleEnv()
    state = env.reset()
    target1 = Q[0][0]
    target2 = Q[1][0]
    done1 = 1.0
    done2 = 1.0
    last_target1 = False
    last_target2 = False
    input("Press Enter to continue...")
    start_time = time.time()
    curr_time = time.time() - start_time
    stop_sequence = False
    while True:
        position1 = np.asarray([state[0]['position'][0], state[0]['position'][1]])
        velocity1 = np.asarray([state[0]['velocity'][0], state[0]['velocity'][1]])
        position2 = np.asarray([state[1]['position'][0], state[1]['position'][1]])
        velocity2 = np.asarray([state[1]['velocity'][0], state[1]['velocity'][1]])
        error1 = target1 - position1
        error2 = target2 - position2
        curr_heading1 = np.arctan2(velocity1[1], velocity1[0])
        curr_heading2 = np.arctan2(velocity2[1], velocity2[0])
        desired_heading1 = np.arctan2(error1[1], error1[0])
        desired_heading2 = np.arctan2(error2[1], error2[0])
        action1 = [speed * done1, (desired_heading1 - curr_heading1) * done1]
        action2 = [speed * done2, (desired_heading2 - curr_heading2) * done2]
        state, reward, done, info = env.step(action1+action2)
        if np.linalg.norm(error1) < 0.1:
            if last_target1:
                done1 = 0.0
            target1 = Q[0][1]
            last_target1 = True
        if np.linalg.norm(error2) < 0.1:
            if last_target2:
                done2 = 0.0
            target2 = Q[1][1]
            last_target2 = True
        if done1 + done2 < 0.1:
            if stop_sequence is False:
                stop_time = time.time()
                stop_sequence = True
            if time.time() - stop_time > 2.0:
                break
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
