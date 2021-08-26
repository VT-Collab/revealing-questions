#!/usr/bin/env python

"""

this script does the following:
(1) loads a trajectory (sequence of joint positions evenly spaced in time)
(2) replays that trajectory over the given time interval

"""

import rospy
import actionlib
import pickle
import sys
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


TO_START_TIME = 3.5     #how long the robot has to get to the first waypoint
AT_START_TIME = 0.5     #how long the robot waits at the first waypoint
AT_END_TIME = 0.25       #how long the robot waits at the last waypoint


class TrajectoryClient(object):

    def __init__(self, duration):
        self.gripper = actionlib.SimpleActionClient(
                '/gripper_controller/gripper_action',
                GripperCommandAction)
        self.gripper.wait_for_server()
        self.client = actionlib.SimpleActionClient(
                '/arm_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory.joint_names = ["shoulder_pan_joint",
                "shoulder_lift_joint", "upperarm_roll_joint",
                "elbow_flex_joint", "forearm_roll_joint",
                "wrist_flex_joint", "wrist_roll_joint"]
        self.duration = duration

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

    def add_waypoint(self, position, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = position
        waypoint.time_from_start = rospy.Duration(time)
        self.goal.trajectory.points.append(waypoint)

    def load_trajectory(self, trajectory):
        self.add_waypoint(trajectory[0], TO_START_TIME)
        self.waypoint_duration = self.duration / (len(trajectory) - 1.0)
        for i, position in enumerate(trajectory):
              time = TO_START_TIME + AT_START_TIME + i * self.waypoint_duration
              self.add_waypoint(position, time)
        self.add_waypoint(trajectory[-1], TO_START_TIME + AT_START_TIME +
              self.duration + AT_END_TIME)

    def send(self, wait):
        self.goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(self.goal)
        self.client.wait_for_result()
        time.sleep(wait)


def main():

    rospy.init_node("play_trajectory")

    while not rospy.is_shutdown():

        mover = TrajectoryClient(duration)
        mover.load_trajectory(data)
        mover.send()
        mover.gripper_open()
        mover.gripper_close()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
