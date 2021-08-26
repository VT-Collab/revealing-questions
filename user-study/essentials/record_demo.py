#!/usr/bin/env python

"""

this script does the following:
(1) send robot to a home position of your choosing
(2) enter the backdrive mode
(3) records joint positions as you backdrive the robot
(4) you can reset to the same start point for another demonstration

"""

import rospy
import actionlib
import pickle
import sys
import time
import pygame
import numpy as np

from power_msgs.srv import BreakerCommand
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import (
      JointState
)


HOME_POSITION = [0.14367080900648516, -1.0402462503531962, -3.257483552703506, -1.7801367945199174, 0.0789840881388051, -0.8353921099200822, -0.00945298474171628]
HOMING_TIME = 4.0


class TrajectoryClient(object):

    def __init__(self):

        self.client = actionlib.SimpleActionClient(
                '/arm_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        self.joint_names = ["shoulder_pan_joint",
                "shoulder_lift_joint", "upperarm_roll_joint",
                "elbow_flex_joint", "forearm_roll_joint",
                "wrist_flex_joint", "wrist_roll_joint"]

    def send(self, position, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = position
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.sleep(time)


class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

    def getKey(self):
        pygame.event.get()
        stop = self.gamepad.get_button(7)
        reset = self.gamepad.get_button(0)
        return stop, reset


class RecordDemonstration(object):

  def __init__(self, steptime):
    self.steptime = rospy.Duration(steptime)
    self.starttime = rospy.Time.now()
    self.lasttime = rospy.Time.now()
    rospy.Subscriber("/joint_states", JointState, self.recorder)
    self.data = []
    self.record = False

  def recorder(self, msg):
    currtime = msg.header.stamp
    position = msg.position
    if self.record:
        if currtime - self.lasttime > self.steptime and len(position) > 10:
            self.lasttime += self.steptime
            arm_position = position[6:13]
            self.data.append(arm_position)

  def start_recording(self):
      self.record = True
      self.lasttime = rospy.Time.now()

  def stop_recording(self):
      self.record = False


def reset_breaker():
    rospy.wait_for_service('/arm_breaker')
    reset_arm_breaker = rospy.ServiceProxy('/arm_breaker', BreakerCommand)
    reset_arm_breaker(False)
    reset_arm_breaker(True)
    rospy.sleep(3.0)


def main():

    savename = sys.argv[1]
    steptime = float(sys.argv[2])
    print(sys.argv[2])
    rospy.init_node("joint_state_recorder")
    mover = TrajectoryClient()
    recorder = RecordDemonstration(steptime)
    joystick = JoystickControl()

    count = 0

    while True:

        savename_number = savename + str(count) + ".pkl"
        mover.send(HOME_POSITION, HOMING_TIME)
        reset_breaker()
        print("ready for a demonstration!")
        recorder.start_recording()

        while True:

            stop, reset = joystick.getKey()
            if stop:
                recorder.stop_recording()
                pickle.dump(recorder.data, open(savename_number, "wb"))
                return True
            if reset:
                recorder.stop_recording()
                pickle.dump(recorder.data, open(savename_number, "wb"))
                recorder.data = []
                count += 1
                break


if __name__ == "__main__":
    main()
