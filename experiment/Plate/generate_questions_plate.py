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
from sensor_msgs.msg import (
      JointState
)

# Bookmarked waypoint of Fetch arm
q1 = [-0.4674811363220215, -0.7673740386962891, -3.141592502593994,
-1.5194079875946045, -0.15684953331947327, 0.4398689270019531, 1.6206507682800293]

q2 = [0.06979608535766602, -0.9334274530410767, 3.141209602355957,
-1.7828691005706787, -0.07286427915096283, 0.6036214828491211, 1.6214181184768677]

q3 = [0.698728084564209, -1.0684177875518799, -3.141592502593994,
-2.114208936691284, -0.12463594228029251, 0.6047718524932861, 1.6206507682800293]


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
        # define limits for feature noises
        self.x = 0
        self.y = 0
        self.h = 0

        self.lower_b_x = 0
        self.upper_b_x = 0.2

        self.lower_b_y = -0.7
        self.upper_b_y = 0

        self.lower_b_h = 0
        self.upper_b_h = 0.3

    def compute_noise(self):
        # self.x = np.random.uniform(self.lower_b_x,self.upper_b_x)
        self.x = np.clip(np.random.normal(0,0.3),self.lower_b_x,self.upper_b_x)

        # self.y = np.random.uniform(self.lower_b_y,self.upper_b_y)
        self.y = np.clip(np.random.normal(-0.3,0.5),self.lower_b_y,self.upper_b_y)

        # self.h = np.random.uniform(self.lower_b_h,self.upper_b_h)
        self.h = np.clip(np.random.normal(0,0.3),self.lower_b_h,self.upper_b_h)
        return [self.x, self.y, self.h]

    def kinematic_tunnel(self, x1, x2, x3, robot):
        set = np.array([x1,x2,x3])
        Q = []
        for i in range(set.shape[0]):
            flag = False
            while flag is False:
                noise = self.compute_noise()
                transformation = robot.dirkin(set[i])
                if i == 0:
                    initial = set[0]
                elif i == 1:
                    transformation[2,3]+= noise[2]
                    initial = set[0]
                else:
                    transformation[0,3] += noise[0]
                    transformation[1,3] += noise[1]
                    transformation[2,3] += noise[2]
                    initial = set[1]
                q_inv = robot.invkin(transformation, initial)
                if q_inv is not None:
                    flag = True
            Q.append(tuple(q_inv.tolist()))
        return Q

    def compute_features(self):
        # height from the table
        f1 = self.h/(max(abs(self.upper_b_h),abs(self.lower_b_h)))
        # distance from the target
        f2_max = np.linalg.norm([max(abs(self.upper_b_x),abs(self.lower_b_x)),
        max(abs(self.upper_b_y),abs(self.lower_b_y))])
        f2 = np.linalg.norm([self.x, self.y])/f2_max

        f = tuple([(f1), (f2)])
        return f

def question(n):
    # this function creates the question dataset
    robot = FetchRobot()
    kin = Kinematics()
    dataset = []
    for _ in range(int(n)):
        trajectory = kin.kinematic_tunnel(q1,q2,q3,robot)
        features = kin.compute_features()
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
        n_questions = 500
        trajectory_set = question(n_questions)

        Question_list = []
        for i in range(len(trajectory_set)/2):
            Question_list.append([trajectory_set[i], trajectory_set[i+1]])
        # create and save paths
        savename = 'soheil/fetch-ws/revealing-questions/Data/Questions/Q_plate.pkl'
        pickle.dump(Question_list, open(savename, "wb"))


if __name__ == "__main__":
    main()
