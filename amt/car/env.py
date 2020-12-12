import os
import numpy as np
import pybullet as p
import pybullet_data
from car import Racecar
from objects import YCBObject, InteractiveObj, RBOObject


class SimpleEnv():

    def __init__(self):
        # create simulation (GUI)
        self.urdfRootPath = pybullet_data.getDataPath()
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)

        # set up camera
        self._set_camera()

        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, 0.0])

        self.offset1 = -2.0
        self.offset2 = +2.0

        # example YCB object
        obj = YCBObject('024_bowl')
        obj.load()
        p.resetBasePositionAndOrientation(obj.body_id, [4, self.offset1, 0.1], [0, 0, 0, 1])
        obj = YCBObject('003_cracker_box')
        obj.load()
        p.resetBasePositionAndOrientation(obj.body_id, [2, -0.4 + self.offset1, 0.1], [0, 0, 0, 1])
        obj = YCBObject('004_sugar_box')
        obj.load()
        p.resetBasePositionAndOrientation(obj.body_id, [1.75, -0.35 + self.offset1, 0.1], [0, 0, 0, 1])
        obj = YCBObject('002_master_chef_can')
        obj.load()
        p.resetBasePositionAndOrientation(obj.body_id, [1.85, -0.42 + self.offset1, 0.1], [0, 0, 0, 1])

        # example YCB object
        obj = YCBObject('024_bowl')
        obj.load()
        p.resetBasePositionAndOrientation(obj.body_id, [4, self.offset2, 0.1], [0, 0, 0, 1])
        obj = YCBObject('003_cracker_box')
        obj.load()
        p.resetBasePositionAndOrientation(obj.body_id, [2, -0.4 + self.offset2, 0.1], [0, 0, 0, 1])
        obj = YCBObject('004_sugar_box')
        obj.load()
        p.resetBasePositionAndOrientation(obj.body_id, [1.75, -0.35 + self.offset2, 0.1], [0, 0, 0, 1])
        obj = YCBObject('002_master_chef_can')
        obj.load()
        p.resetBasePositionAndOrientation(obj.body_id, [1.85, -0.42 + self.offset2, 0.1], [0, 0, 0, 1])

        # load some swarm robots
        self.car1 = Racecar([0, self.offset1, 0])
        self.car2 = Racecar([0, self.offset2, 0])


    def close(self):
        p.disconnect()

    def reset(self):
        self.car1._read_state()
        self.car2._read_state()
        self.car1.state["position"] -= np.asarray([0, self.offset1, 0])
        self.car2.state["position"] -= np.asarray([0, self.offset2, 0])
        return [self.car1.state, self.car2.state]

    def step(self, action):

        # action contains the speed and steering angle
        self.car1.step(speed=action[0], angle=action[1])
        self.car2.step(speed=action[2], angle=action[3])

        # take simulation step
        p.stepSimulation()

        # return next_state, reward, done, info
        self.car1.state["position"] -= np.asarray([0, self.offset1, 0])
        self.car2.state["position"] -= np.asarray([0, self.offset2, 0])
        next_state = [self.car1.state, self.car2.state]
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                     height=self.camera_height,
                                                                     viewMatrix=self.view_matrix,
                                                                     projectionMatrix=self.proj_matrix)
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=-90, cameraPitch=-60,
                                     cameraTargetPosition=[0.75, 0.0, 0.0])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[1.0, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)
