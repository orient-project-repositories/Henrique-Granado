from math import *
import math
import numpy as np
import quaternion
from utils import matrix_multiplication


class System:
    def __init__(self):
        # Objects
        self.eye = Eye()
        self.eye.change_pos(0, 0, 0)
        self.eye.change_scale(0.0753076410465764)

        # Superior and Inferior Rectus
        self.motor1 = Motor()
        self.motor1.change_scale(0.02, 0.1, 0.01)
        self.motor1.change_pos(0.436, 0, -0.0835)

        sr_1 = np.array([0, 90])
        sr_2 = np.array([82.47898517199053, 44.75247673069133])
        self.superior_rectus = Elastic(self.motor1, sr_1, self.eye, sr_2)
        self.superior_rectus.change_color([255, 0, 0], [255, 255, 0])
        self.superior_rectus.add_midpoint(np.array([0.2055, 0.0525, 0]))

        ir_1 = np.array([0, -90])
        ir_2 = np.array([82.47898517199053, -44.75247673069133])
        self.inferior_rectus = Elastic(self.motor1, ir_1, self.eye, ir_2)
        self.inferior_rectus.change_color([255, 0, 0], [255, 255, 0])
        self.inferior_rectus.add_midpoint(np.array([0.2055, -0.0525, 0]))

        # Medial and Lateral Rectus
        self.motor2 = Motor()
        self.motor2.change_scale(0.02, 0.01, 0.14)
        self.motor2.change_pos(0.323, 0, 0)

        lr_1 = np.array([-90, 0])
        lr_2 = np.array([-84.66642703123067, 0])
        self.lateral_rectus = Elastic(self.motor2, lr_1, self.eye, lr_2)
        self.lateral_rectus.change_color([0, 255, 0], [0, 0, 255])
        self.lateral_rectus.change_natural_length(0.24)
        # self.lateral_rectus.add_midpoint(np.array([2.5, 0, -1]))

        mr_1 = np.array([90, 0])
        mr_2 = np.array([84.66642703123067, 0])
        self.medial_rectus = Elastic(self.motor2, mr_1, self.eye, mr_2)
        self.medial_rectus.change_color([0, 255, 0], [0, 0, 255])
        self.medial_rectus.change_natural_length(0.24)
        # self.medial_rectus.add_midpoint(np.array([2.5, 0, 1]))

        # Superior and Inferior Oblique
        self.motor3 = Motor()
        self.motor3.change_scale(0.02, .1, 0.01)
        self.motor3.change_pos(0.436, 0, 0.0835)

        so_1 = np.array([0, 90])
        so_2 = np.array([-82.47898517199053, 44.75247673069133])
        self.superior_oblique = Elastic(self.motor3, so_1, self.eye, so_2)
        self.superior_oblique.change_color([155, 0, 255], [255, 0, 70])
        self.superior_oblique.add_midpoint(np.array([0.2055, 0.0525, 0]))

        io_1 = np.array([0, -90])
        io_2 = np.array([-82.47898517199053, -44.75247673069133])
        self.inferior_oblique = Elastic(self.motor3, io_1, self.eye, io_2)
        self.inferior_oblique.change_color([155, 0, 255], [255, 0, 70])
        self.inferior_oblique.add_midpoint(np.array([0.2055, -0.0525, 0]))

        self.object_list = [self.eye, self.motor1, self.superior_rectus, self.inferior_rectus, self.motor2, self.lateral_rectus, self.medial_rectus, self.motor3, self.superior_oblique,
                            self.inferior_oblique]
        self.motor_list = [self.motor1, self.motor2, self.motor3]
        self.muscle_list = [self.superior_rectus, self.inferior_rectus, self.medial_rectus, self.lateral_rectus, self.superior_oblique, self.inferior_oblique]

        for muscle in self.muscle_list:
            self.eye.add_parent(muscle)

    def do_batch(self):
        for obj in self.object_list:
            obj.do_batch()

    def feed_input(self, step):  # TODO rename this?
        # TODO maybe treat step beforehand
        # step is assumed in radians
        self.motor1.change_orientation(0, 0, step[0])
        self.motor2.change_orientation(0, step[1], 0)
        self.motor3.change_orientation(0, 0, step[2])

    def get_motor_position(self):
        return np.array([self.motor1.get_orientation()[2], self.motor2.get_orientation()[1], self.motor3.get_orientation()[2]])

    def get_output(self):
        return quaternion.as_float_array(self.eye.get_orientation())

    def get_eye_angular_velocity(self):
        return self.eye.get_angular_velocity()

    def do_premovement(self, time, time_step, action):
        self.feed_input(action)
        for i in range(int(time/time_step)):
            self.update(time_step)

    def update(self, dt):
        # pass
        for object in self.motor_list + self.muscle_list + [self.eye]:
            object.update(dt)

    def reset(self):
        for motor in self.motor_list:
            motor.change_orientation(0, 0, 0)
        self.eye.change_orientation(0,0,0)
        self.eye.change_angular_velocity(0, 0, 0)
        for muscle in self.muscle_list:
            muscle.update(0)

    def change_eye_quaternion(self, new_quat):
        self.eye.change_orientation_as_quat(new_quat)

    def stop(self):
        self.eye.change_angular_velocity(0,0,0)

    def get_muscle_list(self):
        return self.muscle_list

    def get_muscle_pairs(self):
        return [[self.superior_rectus, self.inferior_rectus],[self.medial_rectus, self.lateral_rectus],[self.superior_oblique, self.inferior_oblique]]

class Elastic:
    def __init__(self, obj1, loc1, obj2, loc2):
        self.k = -6
        self.natural_length = 0.36

        self.obj1 = obj1
        self.loc1 = loc1  # long and lat
        self.obj2 = obj2
        self.loc2 = loc2

        self.start = None
        self.midpoints = []
        self.end = None

        self.start_color = np.array([0, 255, 0])
        self.end_color = np.array([0, 255, 0])

        # self.batch = pyglet.graphics.Batch()
        self.update(0)

    def add_midpoint(self, point):
        self.midpoints.append(point)
        # self.do_batch()

    def get_insertion2_loc(self):  # TODO change this (quite hateful)
        return self.loc2

    def get_force_direction(self):
        all_points = [self.start]+self.midpoints+[self.end]
        vector = (all_points[-1]-all_points[-2])/np.linalg.norm(all_points[-1]-all_points[-2])
        return vector

    def get_force_magnitude(self):
        return self.k / self.natural_length * max(self.get_total_length() - self.natural_length, 0)

    def get_total_length(self):
        total_length = 0
        points = [self.start] + self.midpoints + [self.end]
        for i in range(len(points) - 1):
            total_length += np.linalg.norm(points[i + 1] - points[i])
        return total_length

    def get_force(self):
        return self.get_force_magnitude()*self.get_force_direction()

    def get_eye_insertion_point(self):
        return self.end

    def get_points(self):
        return [self.start] + self.midpoints + [self.end]

    def get_motor(self):
        return self.obj1

    def change_natural_length(self, length):
        self.natural_length = length

    def change_color(self, start_color, end_color):
        self.start_color = np.array(start_color)
        self.end_color = np.array(end_color)
        # self.do_batch()

    def update(self, dt):
        center1 = self.obj1.get_pos()
        insertion1 = self.obj1.get_point(self.loc1)

        self.start = center1+insertion1

        center2 = self.obj2.get_pos()
        insertion2 = self.obj2.get_point(self.loc2)

        self.end = center2+insertion2
        # self.do_batch()


class Motor:
    def __init__(self):
        # self.image = pic_name
        # self.tex = self.get_tex(self.image)

        self.pos = np.array([0, 0, 0])  # x, y, z
        # self.rot = quaternion.from_euler_angles(0,0,0)  # roll, yaw, pitch
        self.rot = np.array([0, 0, 0])
        self.scale = np.array([1, 1, 1])  # scale_x, scale_y, scale_z

    def get_pos(self):
        return self.pos

    def get_orientation(self):
        return self.rot

    def get_point(self, rot):
        long = radians(rot[0])
        lat = radians(rot[1])
        point = self.scale*[-cos(lat)*cos(long), sin(lat), cos(lat)*sin(long)]
        return quaternion.rotate_vectors(quaternion.from_rotation_vector(self.rot), point)

    def change_pos(self, pos_x, pos_y, pos_z):
        self.pos = np.array([pos_x, pos_y, pos_z])

    def change_orientation(self, roll, yaw, pitch):
        # self.rot = quaternion.from_rotation_vector([roll,yaw,pitch])
        self.rot = np.array([roll, yaw, pitch])

    def change_scale(self,scale_x=1,scale_y=1,scale_z=1):
        self.scale = np.array([scale_x*(scale_x != 0) + (scale_x == 0), scale_y * (scale_y != 0) + (scale_y == 0), scale_z * (scale_z != 0) + (scale_z == 0)])

    def update(self, dt):
        pass


class Eye:
    def __init__(self):
        # Parameters
        self.friction_coefficient = -0.02
        self.static_friction_torque = 0.006

        self.texture_name = 'tex/Eye.png'

        # state
        self.pos = np.array([0, 0, 0])  # x, y, z
        self.rot = quaternion.from_euler_angles([0, 0, 0])  # roll, yaw, pitch
        self.angular_velocity = np.array([0, 0, 0])
        self.scale = np.array([1, 1, 1])  # scale_x, scale_y, scale_z
        self.inertia_tensor = np.array([[0.0004759, 0, 0], [0, 0.0003956, 0], [0, 0, 0.0004316]])*0.1
        self.center_of_mass = [-0.02, 0, 0]
        self.mass = 0.248
        self.gravity = np.array([0, -9.81, 0])

        self.parent_list = []

    def get_pos(self):
        return self.pos

    def get_orientation(self):
        return self.rot

    def get_angular_velocity(self):
        return self.angular_velocity

    def get_point(self, rot):
        long = radians(rot[0])
        lat = radians(rot[1])
        point = self.scale * [-cos(lat) * cos(long), sin(lat), cos(lat) * sin(long)]
        return quaternion.rotate_vectors(self.rot, point)

    def get_current_inertia_tensor(self):
        rotation_matrix = quaternion.as_rotation_matrix(self.rot)
        return matrix_multiplication([np.transpose(rotation_matrix), self.inertia_tensor, rotation_matrix])

    def get_gravity_torque(self):
        rotation_matrix = quaternion.as_rotation_matrix(self.rot)
        current_center_of_mass = np.matmul(rotation_matrix,self.center_of_mass) # TODO (*scale?)
        return np.cross(current_center_of_mass, self.gravity*self.mass)

    def change_pos(self, pos_x, pos_y, pos_z): # TODO change name from change to set
        self.pos = np.array([pos_x, pos_y, pos_z])

    def change_orientation(self, roll, yaw, pitch):
        self.rot = quaternion.from_euler_angles(roll, yaw, pitch)

    def change_orientation_as_quat(self, new_quat):
        self.rot = quaternion.from_float_array(new_quat)

    def change_scale(self, scale):
        self.scale = np.array([scale * (scale != 0) + (scale == 0)]*3)

    def change_angular_velocity(self, w_x, w_y, w_z):
        self.angular_velocity = np.array([w_x, w_y, w_z])

    def add_parent(self, parent):
        self.parent_list.append(parent)

    def update(self, dt):  # TODO make sure dt is taken from somewhere
        torque_elastic = 0
        for parent in self.parent_list:
            this_torque = np.cross(self.get_point(parent.get_insertion2_loc())-self.pos, parent.get_force())
            torque_elastic += this_torque

        torque_friction = self.friction_coefficient*self.angular_velocity
        torque_gravity = 0  # self.get_gravity_torque()
        total_torque = torque_elastic + torque_friction + torque_gravity

        is_moving = (np.linalg.norm(total_torque) > self.static_friction_torque or np.linalg.norm(self.angular_velocity) > 0.00001)
        total_torque = total_torque  * is_moving
        angular_acceleration = np.matmul(np.linalg.inv(self.get_current_inertia_tensor()), total_torque)
        self.angular_velocity = np.add(self.angular_velocity, np.array(angular_acceleration)*dt)* is_moving
        q_angular_velocity = np.quaternion(0, self.angular_velocity[0], self.angular_velocity[1], self.angular_velocity[2])
        q_dot = 0.5*q_angular_velocity*self.rot  # rate_of_change of a quaternion
        self.rot = self.rot + q_dot*dt
        self.rot /= np.linalg.norm(quaternion.as_float_array(self.rot))
