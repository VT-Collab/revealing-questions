import numpy as np
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics

class FetchRobot:

    def __init__(self):
        self.base_link = "torso_lift_link"
        self.end_link = "wrist_roll_link"
        joint_limits_lower = np.array([-92, -87, -180, -129, -180, -125, -180])*np.pi/180
        joint_limits_upper = np.array([92, 70, 180, 129, 180, 125, 180])*np.pi/180
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
        return self.kdl_kin.inverse(pose, q, maxiter=10000, eps=0.001)

    def invkin_search(self, pose, timeout=1.):
        return self.kdl_kin.inverse_search(pose, timeout)


robot = FetchRobot()

q = robot.samplejoint()
qclose = q + np.random.normal(0, 0.1, 7)

pose = robot.dirkin(q)
q_inv_noseed = robot.invkin(pose)
q_inv_close = robot.invkin(pose, qclose)
print(pose)
print(q)
print(q_inv_close)
print(q_inv_noseed)
