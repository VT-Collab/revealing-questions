import numpy as np
import kinpy as kp
import rospy
import torch

from sensor_msgs.msg import (
      JointState
)


class RecordDemonstration(object):

  def __init__(self):
    rospy.Subscriber("/joint_states", JointState, self.recorder)
    self.state = {}

  def recorder(self, msg):
    position = msg.position
    name = msg.name
    if len(position) > 10:
        state = {}
        for idx in range(len(name)):
            state[name[idx]] = position[idx]
        self.state = state

class FetchRobot:

    def __init__(self):
        self.chain = kp.build_chain_from_urdf((open("fetch.urdf")).read())
        self.serial_chain = kp.build_serial_chain_from_urdf((open("fetch.urdf")).read(),'gripper_link')

    def dirkin(self, q):
        pose = self.serial_chain.forward_kinematics(q)
        return pose

    def jacobian(self, q):
        return self.serial_chain.jacobian(q)

    def invkin(self, tg, initial_state):
        jnt_pos = self.serial_chain.inverse_kinematics(tg, initial_state)
        return jnt_pos
