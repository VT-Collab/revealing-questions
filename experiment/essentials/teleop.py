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
import pickle
import sys
import time
import pygame
import numpy as np
import torch


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
from geometry_msgs.msg import(
    TwistStamped,
    Twist
)


HOME_POSITION = [0.14367080900648516, -1.0402462503531962, -3.257483552703506, -1.7801367945199174, 0.0789840881388051, -0.8353921099200822, -0.00945298474171628]
HOMING_TIME = 4.0

STEP_SIZE_L = 0.2
STEP_SIZE_A = np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1


class TrajectoryClient(object):

    def __init__(self):
        self.gripper = actionlib.SimpleActionClient(
                '/gripper_controller/gripper_action',
                GripperCommandAction)
        self.gripper.wait_for_server()
        self.publisher = rospy.Publisher(
                '/arm_controller/cartesian_twist/command',
                TwistStamped, queue_size=1)
        self.publish_rate = rospy.Rate(1)
        self.client = actionlib.SimpleActionClient(
                '/arm_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        self.joint_names = ["shoulder_pan_joint",
                "shoulder_lift_joint", "upperarm_roll_joint",
                "elbow_flex_joint", "forearm_roll_joint",
                "wrist_flex_joint", "wrist_roll_joint"]

    def send(self, twist, time):
        xdot = TwistStamped()
        xdot.twist.linear.x = twist[0]
        xdot.twist.linear.y = twist[1]
        xdot.twist.linear.z = twist[2]
        xdot.twist.angular.x = twist[3]
        xdot.twist.angular.y = twist[4]
        xdot.twist.angular.z = twist[5]
        xdot.header.stamp = rospy.Time.now()
        xdot.header.frame_id = "shoulder_pan_joint"
        self.publisher.publish(xdot)
        rospy.sleep(time)

    def send_joint(self, position, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = position
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.sleep(time)

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


class JointStateListener(object):

  def __init__(self):
    self.position = [0]*7
    rospy.Subscriber("/joint_states", JointState, self.recorder)

  def recorder(self, msg):
    currtime = msg.header.stamp
    position = msg.position
    if len(position) > 10:
        self.position = position[6:13]


class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        z = [z1, z2, z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        gripper_open = self.gamepad.get_button(1)
        gripper_close = self.gamepad.get_button(0)
        return tuple(z), (gripper_open, gripper_close), stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, STEP_SIZE_A * -z[1], STEP_SIZE_A * -z[0], STEP_SIZE_A * -z[2])
        else:
            self.action = (STEP_SIZE_L * -z[1], STEP_SIZE_L * -z[0], STEP_SIZE_L * -z[2], 0, 0, 0)


def main():

    rospy.init_node("endeffector_teleop")
    mover = TrajectoryClient()
    listener = JointStateListener()
    joystick = JoystickControl()
    data = []

    mover.open_gripper()
    mover.send_joint(HOME_POSITION, HOMING_TIME)
    print("ready for inputs!")
    start_time = time.time()
    last_time = 0.0
    step_time = 0.1

    while not rospy.is_shutdown():

        t_curr = time.time() - start_time
        axes, buttons, stop = joystick.getInput()
        if stop:
            #pickle.dump(data, open(filename, "wb"))
            return True
        if buttons[0]:
            mover.open_gripper()
            continue
        if buttons[1]:
            mover.close_gripper()
            continue

        joystick.getAction(axes)
        q_curr = list(listener.position)
        action = joystick.action
        if t_curr - last_time > step_time:
            data.append([t_curr] + q_curr + list(action) + [joystick.toggle])
            print(np.asarray(q_curr))
            last_time = t_curr
        mover.send(action, STEP_TIME)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
