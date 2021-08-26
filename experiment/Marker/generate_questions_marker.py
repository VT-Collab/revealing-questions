"""
This script takes bookmarked joint positions of Fetch arm
(e.g., home position, target position) in the work environment
and combines them in trajectories (questions) with random noises in
end-effector positions
"""

import numpy as np
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import pickle
import tf.transformations as ros_trans
import rospy
import random
from sensor_msgs.msg import (
      JointState
)

# Bookmarked waypoint of Fetch arm
q_n1 = [-0.07439804077148438, -1.1125197410583496, 2.9881949424743652,
-2.260704278945923, -0.17257283627986908, 1.5316798686981201, 0]

q_t1 = [0.3788933753967285, -0.6741846799850464, 2.9881949424743652,
-2.001077890396118, 0.013038462959229946, 1.340315818786621, 0.09088873863220215]

q_t2 = [-0.3267378807067871, -0.6691992282867432, 2.990112543106079,
-1.889864206314087, 0.012271472252905369, 1.202257513999939, 0.09012174606323242]


class RecordDemonstration(object):

    def __init__(self):
        rospy.Subscriber("/joint_states", JointState, self.recorder)
        self.joint_position = None

    def recorder(self, msg):
        currtime = msg.header.stamp
        position = msg.position
        if len(position) > 10:
            self.joint_position = position[6:13]

    def get_curr_position(self):
        while self.joint_position is None:
            x = 1
        return self.joint_position

class FetchRobot:

    def __init__(self):
        self.base_link = "torso_lift_link"
        self.end_link = "wrist_roll_link"
        joint_limits_lower = np.array([-92, -70, -179, -129, -179, -125, -179])*np.pi/180
        joint_limits_upper = np.array([92, 87, 180, 129, 180, 125, 180])*np.pi/180
        self.joint_limits_lower = list(joint_limits_lower)
        self.joint_limits_upper = list(joint_limits_upper)
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        self.kdl_kin.joint_limits_lower = self.joint_limits_lower
        self.kdl_kin.joint_limits_upper = self.joint_limits_upper
        self.kdl_kin.joint_safety_lower = self.joint_limits_lower
        self.kdl_kin.joint_safety_upper = self.joint_limits_upper

    def samplejoint(self):
        q = self.kdl_kin.random_joint_angles()
        return tuple(q)

    def dirkin(self, q):
        pose = np.asarray(self.kdl_kin.forward(q))
        return pose

    def jacobian(self, q):
        return self.kdl_kin.jacobian(q)

    def invkin(self, pose, q=None):
        return self.kdl_kin.inverse(pose, q, maxiter=20000, eps=0.0001)

    def invkin_search(self, pose, timeout=1.):
        return self.kdl_kin.inverse_search(pose, timeout)

class Kinematics:

    def __init__(self):
        self.robot = FetchRobot()
        self.y = 0
        self.lower_b_h = 0
        self.upper_b_h = 0.4
        self.tar_loc = 1

    def option_insertion(self, traj, target):
        ins_at_1 = 1
        traj[ins_at_1:ins_at_1] = [target]
        return traj

    def random_option(self, opt1, opt2):
        pair = [opt1, opt2]
        choice = random.choice(pair)
        index = pair.index(choice)
        return index, choice

    def build_trajectory(self):
        set_main = [q_n1, q_n1]

        vote_tar, target = self.random_option(q_t1, q_t2)
        vote_ang, angle = self.random_option(0, -3)
        target[-1] = angle
        trajectory = self.option_insertion(set_main, target)
        return vote_tar, vote_ang, np.array(trajectory)

    def kinematic_tunnel(self, set):
        # self.h = np.random.uniform(self.lower_b_h,self.upper_b_h)
        self.h = np.clip(np.random.normal(0,0.4),self.lower_b_h,self.upper_b_h)

        transformation = self.robot.dirkin(set[self.tar_loc])
        flag = False
        while flag is False:
            transformation[2,3] += self.h
            initial = q_n1
            q_inv = self.robot.invkin(transformation, initial)
            if q_inv is not None:
                flag = True
            set[self.tar_loc] = q_inv
            Q = [tuple(sub) for sub in set]
        return Q

    def compute_features(self, opt1, opt2):
        # Must pick the right target
        if opt1 == 1:
            f1 = 1.0
        else:
            f1 = 0.0
        # Must hold the marker in the right angle
        if opt2 == 1:
            f2 = 1.0
        else:
            f2 = 0.0
        # Must put a mark on the right target
        f3 = abs(self.h/(max(abs(self.upper_b_h),abs(self.lower_b_h))))

        f = tuple([(f1), (f2), (f3)])
        return f

def question(n):
    # this function creates the question dataset
    kin = Kinematics()
    dataset = []
    for _ in range(int(n)):
        vote_tar, vote_ang, joint_path = kin.build_trajectory()
        trajectory = kin.kinematic_tunnel(joint_path)
        features = kin.compute_features(vote_tar, vote_ang)
        trajectory.extend(features)
        dataset.append(trajectory)
    return dataset


def main():
    record = RecordDemonstration()
    rospy.init_node("joint_state_recorder")

    toggle = False

    if toggle is True:
        print(record.get_curr_position())
    else:
        n_questions = 200
        trajectory_set = question(n_questions)

        Question_list = []
        for i in range(len(trajectory_set)/2):
            Question_list.append([trajectory_set[i], trajectory_set[i+1]])
        # create and save paths
        savename = 'soheil/fetch-ws/revealing-questions/Data/Questions/Q_marker.pkl'
        pickle.dump(Question_list, open(savename, "wb"))


if __name__ == "__main__":
    main()
