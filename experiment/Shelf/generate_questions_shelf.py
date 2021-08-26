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
q_n1 = [-0.015723705291748047, -0.8559613227844238, 3.134690284729004,
-2.300971269607544, 0.015723302960395813, 1.546252727508545, 3.10209321975708]

q_ob1 = [-0.07746601104736328, -0.296441912651062, 2.8946220874786377,
-1.0653495788574219, 0.06941263377666473, 0.7209711074829102, -2.9970149993896484]

q_ob2 = [0.28225231170654297, -0.29605841636657715, 2.894238233566284,
-1.1788642406463623, 0.06864564120769501, 0.9023642539978027, -2.9970149993896484]

q_n2 = [0.042951107025146484, -1.2034082412719727, 2.1617629528045654,
-1.971165418624878, 1.112903118133545, 0.7612380981445312, 2.603165864944458]

q_t1 = [-0.4567427635192871, -0.28148555755615234, 2.157927989959717,
-1.4243011474609375, 0.9836652278900146, 0.5794613361358643, -3.0288448333740234]

q_t2 = [-0.5123496055603027, -0.8916263580322266, 2.1590781211853027,
-1.265150547027588, 0.7696748971939087, 0.27573299407958984, 2.919165849685669]


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
        self.lower_b_y = 0
        self.upper_b_y = 0.2
        self.ins_at_2 = 5

    def option_insertion(self, traj, object, target):
        ins_at_1 = 2
        ins_at_2 = self.ins_at_2
        traj[ins_at_1:ins_at_1] = [object]
        traj[ins_at_2:ins_at_2] = [target]
        return traj

    def random_option(self, opt1, opt2):
        pair = [opt1, opt2]
        choice = random.choice(pair)
        index = pair.index(choice)
        return index, choice

    def build_trajectory(self):
        set_main = [q_n2, q_n1, q_n1, q_n2]
        vote_obj, object = self.random_option(q_ob1, q_ob2)
        vote_tar, target = self.random_option(q_t1, q_t2)
        trajectory = self.option_insertion(set_main, object, target)
        return vote_obj, vote_tar, np.array(trajectory)

    def kinematic_tunnel(self, set):
        self.y = random.choice([self.lower_b_y,self.upper_b_y])
        # self.y = np.random.uniform(self.lower_b_y,self.upper_b_y)
        # self.y = np.clip(np.random.normal(0,0.1),self.lower_b_y,self.upper_b_y)
        transformation = self.robot.dirkin(set[self.ins_at_2])
        flag = False
        while flag is False:
            transformation[1,3] += self.y
            initial = q_n2
            q_inv = self.robot.invkin(transformation, initial)
            if q_inv is not None:
                flag = True
            set[self.ins_at_2] = q_inv
            Q = [tuple(sub) for sub in set]
        return Q

    def compute_features(self, opt1, opt2):
        # Must pick the right object
        if opt1 == 1:
            f1 = 1.0
        else:
            f1 = 0.0
        # Must pick the right shelf
        if opt2 == 1:
            f2 = 1.0
        else:
            f2 = 0.0
        # Must drop the object inside the self
        f3 = abs(self.y/(max(abs(self.upper_b_y),abs(self.lower_b_y))))

        f = tuple([(f1), (f2), (f3)])
        return f

def question(n):
    # this function creates the question dataset
    kin = Kinematics()
    dataset = []
    for _ in range(int(n)):
        opt1, opt2, joint_path = kin.build_trajectory()
        trajectory = kin.kinematic_tunnel(joint_path)
        features = kin.compute_features(opt1, opt2)
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
        n_questions = 150
        trajectory_set = question(n_questions)

        Question_list = []
        for i in range(len(trajectory_set)/2):
            Question_list.append([trajectory_set[i], trajectory_set[i+1]])
        # create and save paths
        savename = 'soheil/fetch-ws/revealing-questions/Data/Questions/Q_shelf.pkl'
        pickle.dump(Question_list, open(savename, "wb"))


if __name__ == "__main__":
    main()
