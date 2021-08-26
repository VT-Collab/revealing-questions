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

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)


TO_START_TIME = 4.0     #how long the robot has to get to the first waypoint
AT_START_TIME = 2.0     #how long the robot waits at the first waypoint
AT_END_TIME = 0.25       #how long the robot waits at the last waypoint


class TrajectoryClient(object):

    def __init__(self, duration):

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

    def add_waypoint(self, position, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = position
        waypoint.time_from_start = rospy.Duration(time)
        self.goal.trajectory.points.append(waypoint)
        # print(position)

    def load_trajectory(self, filename):
        traj_file = open(filename, 'rb')
        trajectory = pickle.load(traj_file)
        traj_file.close()
        self.add_waypoint(trajectory[0], TO_START_TIME)
        self.waypoint_duration = self.duration / (len(trajectory) - 1.0)
        for i, position in enumerate(trajectory):
              time = TO_START_TIME + AT_START_TIME + i * self.waypoint_duration
              self.add_waypoint(position, time)
        self.add_waypoint(trajectory[-1], TO_START_TIME + AT_START_TIME +
              self.duration + AT_END_TIME)

    def send(self):
        self.goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(self.goal)


def main():

    print(torch.FloatTensor([0.1]))
    filename = sys.argv[1]
    duration = float(sys.argv[2])
    rospy.init_node("play_trajectory")
    traj = TrajectoryClient(duration)
    traj.load_trajectory(filename)
    traj.send()

if __name__ == "__main__":
    main()
