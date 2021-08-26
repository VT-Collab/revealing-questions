#!/usr/bin/env python

"""
this script does the following:
(1) send robot to a home position of your choosing
(2) you can teleop the end effector position and orientation
(3) you can open and close the gripper
(4) robot records its joint position, which can be recorded
"""

import rospy
import actionlib
import sys
import time
import pygame
import numpy as np
import torch
import time

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
    GripperCommand
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import (
    JointState
)


class GripperClient(object):

    def __init__(self):
        self.gripper = actionlib.SimpleActionClient(
                '/gripper_controller/gripper_action',
                GripperCommandAction)
        self.gripper.wait_for_server()

    def open_gripper(self):
        command = GripperCommand
        command.max_effort = 60
        command.position = 0.1
        waypoint = GripperCommandGoal
        waypoint.command = command
        self.gripper.send_goal(waypoint)

    def close_gripper(self):
        command = GripperCommand
        command.max_effort = 60
        command.position = 0.0
        waypoint = GripperCommandGoal
        waypoint.command = command
        self.gripper.send_goal(waypoint)


def main():

    rospy.init_node("endeffector_teleop")
    mover = GripperClient()

    while not rospy.is_shutdown():

        mover.close_gripper()
        mover.open_gripper()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
