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
    self.position = [0]*8

  def recorder(self, msg):
    currtime = msg.header.stamp
    position = msg.position
    name  = msg.name
    if len(position) > 10:
        self.position[0] = position[2]
        self.position[1:] = position[6:]
        # print(self.position)
        state = {}
        for idx in range(len(name)):
            state[name[idx]] = position[idx]
        self.state = state



class FetchRobot:

    def __init__(self):
        self.chain = kp.build_chain_from_urdf((open("fetch.urdf")).read())

    def dirkin(self, q):
        pose = self.chain.forward_kinematics(q)
        return pose


rospy.init_node("kinematics")
robot = FetchRobot()
record = RecordDemonstration()

while not rospy.is_shutdown():
    joint_angles = list(record.state.values())
    print(joint_angles[-7:],'\n')
    pose = robot.dirkin(record.state)
    # print(pose["gripper_link"])


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
