import os
import numpy as np
import pybullet as p
import pybullet_data


class Racecar():

    def __init__(self, basePosition=[0,0,0.1]):
        self.urdfRootPath = pybullet_data.getDataPath()
        self.car = p.loadURDF(os.path.join(self.urdfRootPath,"racecar/racecar.urdf"), basePosition=basePosition)
        self.inactive_wheels = [5, 7]
        self.wheels = [2, 3]
        self.steering = [4, 6]
        for wheel in self.inactive_wheels:
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        self.state = {}


    """functions that environment should use"""

    # set the car's speed and steering angle
    def step(self, speed, angle, maxForce=1000.0):

        # velocity control
        self._velocity_control(speed, angle, maxForce)

        # update robot state measurement
        self._read_state()


    """internal functions"""

    def _read_state(self):
        state = p.getLinkState(self.car, 0, computeLinkVelocity=1)
        position = list(state[0])
        velocity = list(state[6])
        self.state['position'] = np.asarray(position)
        self.state['velocity'] = np.asarray(velocity)

    def _velocity_control(self, speed, angle, maxForce):
        for wheel in self.wheels:
            p.setJointMotorControl2(self.car,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=speed,
                                    force=maxForce)
        for steer in self.steering:
            p.setJointMotorControl2(self.car, steer, p.POSITION_CONTROL, targetPosition=angle)
