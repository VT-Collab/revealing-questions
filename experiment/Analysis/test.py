import numpy as np
import pickle
import optimal_question as opt
import sys


def load_result_qs(method = 'ig',user='0', data = 'qs'):
    filename = "../Data/Study2/user"+ user + "/plate_" + method + "_" + data + ".pkl"
    result = pickle.load(open(filename, "rb"))
    return result


def load_result_ans(method = 'ig',user='0', data = 'ans'):
    filename = "../Data/Study2/user"+ user + "/plate_" + method + "_" + data + ".pkl"
    result = pickle.load(open(filename, "rb"))
    return result

Qstar = load_result_qs()
q = load_result_ans()

# Ask the user if the robot is ready to deploy?

for idx in range(len(Qstar)):
    print("Question: ",'\n', Qstar[idx],'\n')
    print("Answer: ",'\n', q[idx],'\n')

    input("---Next? ")
