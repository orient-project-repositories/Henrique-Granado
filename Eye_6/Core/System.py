import pyglet
from pyglet.gl import *
from math import *
import math
import numpy as np
import quaternion
from utils import matrix_multiplication


class System:
    def __init__(self):
        self.scale = 0.04

        self.object_list = []
        self.motor_list = []

        self.eye = Eye()
        self.eye.change_pos(0, 0, 0)
        self.eye.change_scale(self.scale)

        radius = .024
        ei = np.array([2, -1.4, -40])  # eye insertion point
        hi = np.array([-100.1, -7.8, -40.7])*10**-3  # head insertion point
        self.spindle_ir = Spindle(self.eye, np.degrees([np.arctan2(ei[1], ei[0]), np.arctan(ei[2]/np.linalg.norm(ei))]))
        self.spindle_ir.change_color([0, 255, 0], [0, 255, 0])
        self.spindle_ir.change_radius(radius)
        self.spindle_ir.add_midpoint(hi)
        self.spindle_ir.change_pos(hi+np.array([-0.2, -radius, 0]))
        self.eye.add_parent(self.spindle_ir)
        self.motor_list.append(self.spindle_ir)
        self.object_list.append(self.spindle_ir)

        ei = np.array([2, 0.8, 40])  # eye insertion point
        hi = np.array([-100.1, 14.9, -40.7])*10**-3  # head insertion point
        self.spindle_sr = Spindle(self.eye, np.degrees([np.arctan2(ei[1], ei[0]), np.arctan(ei[2]/np.linalg.norm(ei))]))
        self.spindle_sr.change_color([0, 0, 255], [0, 0, 255])
        self.spindle_sr.change_radius(radius)
        self.spindle_sr.add_midpoint(hi)
        self.spindle_sr.change_pos(hi+np.array([-0.4, -radius, 0]))
        self.eye.add_parent(self.spindle_sr)
        self.motor_list.append(self.spindle_sr)
        self.object_list.append(self.spindle_sr)

        ei = np.array([-7.7, -39.3, 0])  # eye insertion point
        hi = np.array([-100.1, 3.5, -51.6])*10**-3  # head insertion point
        self.spindle_mr = Spindle(self.eye, np.degrees([np.arctan2(ei[1], ei[0]), np.arctan(ei[2]/np.linalg.norm(ei))]))
        self.spindle_mr.change_color([255, 0, 0], [255, 0, 0])
        self.spindle_mr.change_radius(radius)
        self.spindle_mr.add_midpoint(hi)
        self.spindle_mr.change_pos(hi+np.array([-0.6, -radius, 0]))
        self.eye.add_parent(self.spindle_mr)
        self.motor_list.append(self.spindle_mr)
        self.object_list.append(self.spindle_mr)

        ei = np.array([-7.7, 39.3, 0])  # eye insertion point
        hi = np.array([-100.1, 3.5, -29.6])*10**-3  # head insertion point
        self.spindle_lr = Spindle(self.eye, np.degrees([np.arctan2(ei[1], ei[0]), np.arctan(ei[2]/np.linalg.norm(ei))]))
        self.spindle_lr.change_color([255, 0, 255], [255, 0, 255])
        self.spindle_lr.change_radius(radius)
        self.spindle_lr.add_midpoint(hi)
        self.spindle_lr.change_pos(hi+np.array([-0.8, -radius, 0]))
        self.eye.add_parent(self.spindle_lr)
        self.motor_list.append(self.spindle_lr)
        self.object_list.append(self.spindle_lr)

        ei = np.array([11.2, -1.4, -38.3])  # eye insertion point
        hi = np.array([45, -62, -37.5])*10**-3  # head insertion point
        self.spindle_io = Spindle(self.eye, np.degrees([np.arctan2(ei[1], ei[0]), np.arctan(ei[2]/np.linalg.norm(ei))]))
        self.spindle_io.change_color([255, 255, 0], [255, 255, 0])
        self.spindle_io.change_radius(radius)
        self.spindle_io.add_midpoint(hi)
        self.spindle_io.change_pos(hi+np.array([-1, -radius, 0]))
        self.eye.add_parent(self.spindle_io)
        self.motor_list.append(self.spindle_io)
        self.object_list.append(self.spindle_io)

        ei = np.array([11.8, -1.2, 38.3])  # eye insertion point
        hi = np.array([45, 62, -37.5])*10**-3  # head insertion point
        self.spindle_so = Spindle(self.eye, np.degrees([np.arctan2(ei[1], ei[0]), np.arctan(ei[2]/np.linalg.norm(ei))]))
        self.spindle_so.change_color([0, 255, 255], [0, 255, 255])
        self.spindle_so.change_radius(radius)
        self.spindle_so.add_midpoint(hi)
        self.spindle_so.change_pos(hi+np.array([-1, -radius, 0]))
        self.eye.add_parent(self.spindle_so)
        self.motor_list.append(self.spindle_so)
        self.object_list.append(self.spindle_so)

        # ei = np.array([11.8, -1.2, 38.1])  # eye insertion point
        # hi = np.array([-45, 37.5, 62])*10**-3  # head insertion point
        # self.spindle_e = Spindle(self.eye, np.degrees([np.arctan2(ei[1], ei[0]), np.arctan(ei[2]/np.linalg.norm(ei))]))
        # self.spindle_e.change_radius(radius)
        # self.spindle_e.add_midpoint(hi)
        # self.spindle_e.change_pos(hi+np.array([1, -radius, 0]))
        # self.eye.add_parent(self.spindle_e)
        # self.motor_list.append(self.spindle_e)
        # self.object_list.append(self.spindle_e)

        self.object_list.append(self.eye)

    def do_batch(self):
        for obj in self.object_list:
            obj.do_batch()

    def feed_input(self, step):  # TODO rename this?
        # TODO maybe treat step beforehand
        # step is assumed in radians
        self.spindle_ir.rotate_spindle(step[0])
        self.spindle_sr.rotate_spindle(step[1])
        self.spindle_mr.rotate_spindle(step[2])
        self.spindle_lr.rotate_spindle(step[3])
        self.spindle_io.rotate_spindle(step[4])
        self.spindle_so.rotate_spindle(step[5])
    def get_motor_position(self):
        return np.array([self.motor1.get_orientation()[2], self.motor2.get_orientation()[1], self.motor3.get_orientation()[2]])

    def get_output(self):
        return quaternion.as_float_array(self.eye.get_orientation())

    def do_premovement(self, time, time_step, action):
        self.feed_input(action)
        for i in range(int(time/time_step)):
            self.update(time_step)

    def update(self, dt):
        for object in self.motor_list + [self.eye]:
            object.update(dt)

    def render(self, only_horizontal = False):
        scale = 1/self.scale
        glPushMatrix()
        glScalef(scale, scale, scale)
        for i, obj in enumerate(self.object_list):
            if only_horizontal and i not in [0,4,5,6]:
                continue
            obj.render()
        glPopMatrix()

    def reset(self):
        for motor in self.motor_list:
            motor.change_orientation(0, 0, 0)
        self.eye.change_orientation(0,0,0)
        self.eye.change_angular_velocity(0, 0, 0)
        for muscle in self.muscle_list:
            muscle.update(0)


class Elastic2:
    def __init__(self, obj1, obj2, loc2):
        self.k = -6
        self.natural_length = 0.36

        self.obj1 = obj1
        self.obj2 = obj2
        self.loc2 = loc2 # long and lat

        self.start = None
        self.midpoints = []
        self.end = None

        self.start_color = np.array([0, 255, 0])
        self.end_color = np.array([0, 255, 0])

        # self.batch = pyglet.graphics.Batch()
        self.update()
        self.update_start()

    def do_batch(self):
        pass

    def add_midpoint(self, point):
        self.midpoints.append(point)

    def do_draw(self):
        # self.batch = pyglet.graphics.Batch()
        n_points = len(self.midpoints) + 2

        t_l = self.get_total_length()

        point_list = [self.start] + self.midpoints + [self.end]
        color_list = list(self.start_color)
        tl_so_far = 0
        for i in range(len(point_list) - 1):
            tl_so_far += np.linalg.norm(point_list[i + 1] - point_list[i])
            percent = tl_so_far/t_l

            color_list += list(np.int32(self.end_color*percent+self.start_color*(1-percent)))

        points = np.concatenate([self.start] + self.midpoints + [self.end], axis=0)

        pyglet.graphics.draw(n_points, GL_LINE_STRIP, ('v3f', points), ('c3B', color_list))

        for point in self.midpoints:
            pyglet.graphics.draw(1, GL_POINTS, ('v3f', point), ('c3B', [0, 0, 0]))
        pyglet.graphics.draw(1, GL_POINTS, ('v3f', self.end), ('c3B', self.end_color))

    def get_insertion2_loc(self):  # TODO change this (quite hateful)
        return self.loc2

    def get_force_direction(self):
        all_points = [self.start]+self.midpoints+[self.end]
        vector = (all_points[-1]-all_points[-2])/np.linalg.norm(all_points[-1]-all_points[-2])
        return vector

    def get_force_magnitude(self):
        return self.k / self.natural_length * max(self.get_total_length() - self.natural_length, 0)

    def get_unextended_length(self):
        total_length = 0
        points = [self.start] + self.midpoints + [self.end]
        for i in range(len(points) - 1):
            total_length += np.linalg.norm(points[i + 1] - points[i])
        return total_length

    def get_total_length(self):
        total_length = self.obj1.get_elastic_extension()+self.get_unextended_length()
        return total_length

    def get_force(self):
        return self.get_force_magnitude()*self.get_force_direction()

    def get_eye_insertion_point(self):
        return self.end

    def get_second_point(self):
        return (self.midpoints+[self.end])[0]

    def change_natural_length(self, length):
        self.natural_length = length

    def change_color(self, start_color, end_color):
        self.start_color = np.array(start_color)
        self.end_color = np.array(end_color)
        # self.do_batch()

    def update(self):
        center2 = self.obj2.get_pos()
        insertion2 = self.obj2.get_point(self.loc2)
        self.end = center2+insertion2

    def update_start(self):
        center1 = self.obj1.get_pos()
        next_point = self.get_second_point()

        insertion1 = self.obj1.get_point(next_point)

        self.start = center1+insertion1

    def render(self):
        glLineWidth(4)
        self.do_draw()
        # self.batch.draw()


class Floor:
    def __init__(self):
        self.batch = pyglet.graphics.Batch()

        self.do_batch()

    def do_batch(self):
        floor_size = 50
        square_size = 5
        y_level = -6
        for i in range(-floor_size,floor_size+1, square_size):
            for j in range(-floor_size, floor_size+1, square_size):
                verts = [i,y_level,j, i+square_size,y_level,j, i+square_size,y_level,j+square_size, i,y_level,j+square_size]
                c = 100*((i+j)%(2*square_size) == 0)
                color = [60+c,30+c,50+c,255]*4
                self.batch.add(4, GL_QUADS, None, ('v3f', verts), ('c4B', color))

    def update(self, dt):
        pass

    def render(self, extra):
        self.batch.draw()


class Motor2:
    def __init__(self):
        self.texture_name = "tex/Spindle.png"
        self.batch = pyglet.graphics.Batch()

        self.pos = np.array([0, 0, 0])  # x, y, z
        self.spin = 0
        self.rot = np.array([0, 0, 0])
        self.scale = np.array([1, 1, 1])  # scale_x, scale_y, scale_z

        self.do_batch()

    def get_pos(self):
        return self.pos

    def get_orientation(self):
        return self.rot

    def get_point(self, connecting_point):
        x1, y1, z1 = connecting_point
        x0, y0, z0 = self.pos

        delta_g = np.sqrt((x1-x0)**2+(z1-z0)**2)
        delta_h = y1-y0

        theta = -np.arccos(self.scale[1]/np.sqrt(delta_g**2+delta_h**2))-np.arctan2(delta_h,delta_g)

        return np.matmul(-np.array([[np.cos(self.rot[1]), 0, np.sin(self.rot[1])],[0,1,0],[np.sin(-self.rot[1]), 0, np.cos(self.rot[1])]]),self.scale*np.array([0, np.sin(theta), np.cos(theta)]))

    def get_elastic_extension(self):
        return max(self.rot[1]*self.spin, 0)

    def change_pos(self, pos_x, pos_y, pos_z):
        self.pos = np.array([pos_x, pos_y, pos_z])

    def change_orientation(self, roll, yaw, pitch):
        # self.rot = quaternion.from_rotation_vector([roll,yaw,pitch])
        self.rot = np.array([roll, yaw, pitch])

    def change_radius(self, new_radius):
        if new_radius <= 0:
            return
        self.scale = np.array([new_radius*0.1, new_radius, new_radius])
        self.do_batch()

    def change_spin(self, new_spin):
        self.spin = new_spin

    def do_batch(self):
        self.batch = pyglet.graphics.Batch()
        phi = (np.sqrt(5)-1)/2
        #wheel
        outter_wheel_start = -self.scale[0]
        outter_wheel_end = self.scale[0]
        inner_wheel_start = -self.scale[0]*phi
        inner_wheel_end = self.scale[0]*phi
        radius = self.scale[1]
        outter_radius = radius*1.1

        wheel_base1 = [outter_wheel_start, 0,0]
        texc1 = [0.5,0.5]
        wheel_base2 = [outter_wheel_end, 0, 0]
        texc2 = [0.5, 0.5]

        outter_radius_side1 = []
        outter_inner_radius1 = []
        inner_side = []
        inner_outter_radius2 = []
        outter_radius_side2 = []

        for theta in np.linspace(0, 2*np.pi, 720):
            outter_stuff = [outter_radius*np.cos(theta), outter_radius*np.sin(theta)]
            inner_stuff = [radius * np.cos(theta), radius * np.sin(theta)]
            wheel_base1 += [outter_wheel_start]+outter_stuff
            outter_radius_side1 += [outter_wheel_start]+outter_stuff+[inner_wheel_start]+outter_stuff
            outter_inner_radius1 += [inner_wheel_start]+outter_stuff+[inner_wheel_start]+inner_stuff
            inner_side += [inner_wheel_start]+inner_stuff+[inner_wheel_end]+inner_stuff
            inner_outter_radius2 += [inner_wheel_end]+inner_stuff+[inner_wheel_end]+outter_stuff
            outter_radius_side2 += [inner_wheel_end]+outter_stuff+[outter_wheel_end]+outter_stuff
            wheel_base2 += [outter_wheel_end]+outter_stuff

            texc1 += [0.5*np.cos(theta+np.pi/2)+0.5, 0.5*np.sin(theta+np.pi/2)+0.5]
            texc2 += [0.5*np.cos(theta+np.pi/2)+0.5, 0.5*np.sin(theta+np.pi/2)+0.5]

        for group, texc in zip([wheel_base1, wheel_base2],[texc1, texc2]):
            self.batch.add(len(group)//3, GL_TRIANGLE_FAN, get_tex(self.texture_name), ('v3f', group), ('t2f', texc))

        for group, c in zip([outter_radius_side1, outter_inner_radius1, inner_side, inner_outter_radius2, outter_radius_side2],[70, 50, 30, 50, 70]):
            self.batch.add(len(group) // 3, GL_QUAD_STRIP, pyglet.graphics.Group(), ('v3f', group), ('c3B', [c, c, c] * (len(group) // 3)))


        #axis thing
        axis_radius = self.scale[0]*phi
        start = -self.scale[0]*1.75
        end = self.scale[0]*1.75
        ## bottom and top
        top = [end,0,0]
        bottom = [start,0,0]
        side = []
        for theta in np.linspace(0, 2*np.pi, 7):
            top_point = [end, axis_radius*np.cos(theta), axis_radius*np.sin(theta)]
            bottom_point = [start, axis_radius * np.cos(theta), axis_radius * np.sin(theta)]
            top += top_point
            bottom += bottom_point
            side += top_point+[start]+top_point[1:]

        self.batch.add(len(bottom) // 3, GL_TRIANGLE_FAN, pyglet.graphics.Group(), ('v3f', bottom), ('c3B', [0, 0, 0] * (len(bottom)//3)))
        self.batch.add(len(side) // 3, GL_QUAD_STRIP, pyglet.graphics.Group(), ('v3f', side), ('c3B', [0, 0, 0] * (len(side)//3)))
        self.batch.add(len(top)//3, GL_TRIANGLE_FAN, pyglet.graphics.Group(), ('v3f', top), ('c3B', [0,0,0]*(len(top)//3)))
        # self.batch.add(6, GL_LINES, None, ('v3f', np.array([-1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1])*min(self.scale*1.75)), ('c3B', [0, 0, 0]*6))

    def render(self):
        glLineWidth(8)
        euler = self.rot
        angle = np.linalg.norm(self.rot)*180/np.pi

        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(angle, euler[0], euler[1], euler[2])
        glRotatef(np.degrees(self.spin), 1, 0, 0)
        self.batch.draw()
        glPopMatrix()


class Eye:
    def __init__(self):
        # Parameters
        self.friction_coefficient = -0.02
        self.static_friction_torque = 0.006

        self.texture_name = 'tex/Eye.png'
        self.batch = pyglet.graphics.Batch()

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

        self.do_batch()

    def do_batch(self):
        tex = get_tex(self.texture_name)
        self.batch = pyglet.graphics.Batch()
        step = 10
        for lat in range(-90, 90, step):
            verts = []
            texc = []
            for lon in range(0, 361, step):
                x = -self.scale[0] * cos(radians(lat)) * cos(radians(lon))
                y = self.scale[1] * sin(radians(lat))
                z = self.scale[2] * cos(radians(lat)) * sin(radians(lon))
                s = (lon) / 360.0
                t = (lat + 90) / 180.0
                verts += [x, y, z]
                texc += [s, t]
                x = -self.scale[0] * cos(radians(lat+step)) * cos(radians(lon))
                y = self.scale[1] * sin(radians(lat+step))
                z = self.scale[2] * cos(radians(lat+step)) * sin(radians(lon))
                s = (lon) / 360.0  # (lon + 180) / 360.0
                t = ((lat + step) + 90) / 180.0
                verts += [x, y, z]
                texc += [s, t]

            self.batch.add(len(verts)//3, GL_TRIANGLE_STRIP, tex, ('v3f', verts), ('t2f', texc))

        points = list(np.add(self.scale,-np.array([0.1, 1, 1])*self.scale))
        points += list(np.add(self.scale,-np.array([-0.5,1,1])*self.scale))
        self.batch.add(2, GL_LINE_STRIP, None, ('v3f', points), ('c3B', 2 * [90, 250, 200]))

    def get_pos(self):
        return self.pos

    def get_orientation(self):
        return self.rot

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
        current_center_of_mass = np.matmul(rotation_matrix,self.center_of_mass)
        return np.cross(current_center_of_mass, self.gravity*self.mass)

    def change_pos(self, pos_x, pos_y, pos_z): # TODO change name from change to set
        self.pos = np.array([pos_x, pos_y, pos_z])

    def change_orientation(self, roll, yaw, pitch):
        self.rot = quaternion.from_rotation_vector([roll, yaw, pitch])

    def change_scale(self, scale):
        self.scale = np.array([scale * (scale != 0) + (scale == 0)]*3)
        self.do_batch()

    def change_angular_velocity(self, w_x, w_y, w_z):
        self.angular_velocity = np.array([w_x, w_y, w_z])

    def add_parent(self, parent):
        self.parent_list.append(parent)

    def render(self):
        glLineWidth(4)

        euler = quaternion.as_rotation_vector(self.rot)*180/math.pi
        angle = np.linalg.norm(euler)

        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(angle, euler[0], euler[1], euler[2])
        self.batch.draw()
        glPopMatrix()

    def update(self, dt):  # TODO make sure dt is taken from somewhere
        torque_elastic = 0
        for parent in self.parent_list:
            this_torque = np.cross(self.get_point(parent.get_insertion2_loc())-self.pos, parent.get_force())
            torque_elastic += this_torque

        torque_friction = self.friction_coefficient*self.angular_velocity
        torque_gravity = 0  # self.get_gravity_torque()
        total_torque = torque_elastic + torque_friction + torque_gravity

        is_moving = (np.linalg.norm(total_torque) > self.static_friction_torque or np.linalg.norm(self.angular_velocity) > 0.00001)
        total_torque = total_torque * is_moving
        angular_acceleration = np.matmul(np.linalg.inv(self.get_current_inertia_tensor()), total_torque)
        self.angular_velocity = np.add(self.angular_velocity, np.array(angular_acceleration)*dt)* is_moving
        q_angular_velocity = np.quaternion(0, self.angular_velocity[0], self.angular_velocity[1], self.angular_velocity[2])
        q_dot = 0.5*q_angular_velocity*self.rot  # rate_of_change of a quaternion
        self.rot = self.rot + q_dot*dt
        self.rot /= np.linalg.norm(quaternion.as_float_array(self.rot))


class Spindle:
    def __init__(self, obj, loc):
        self.motor = Motor2()
        self.elastic = Elastic2(self.motor, obj, loc)

        self.adjust_motor()

    def change_pos(self, new_pos):
        self.motor.change_pos(new_pos[0], new_pos[1], new_pos[2])
        self.adjust_motor()

    def change_radius(self, new_radius):
        self.motor.change_radius(new_radius)
        self.adjust_motor()

    def change_color(self, start_color, end_color):
        self.elastic.change_color(start_color, end_color)

    def rotate_spindle(self, new_angle):
        self.motor.change_spin(new_angle)

    def add_midpoint(self, point):
        self.elastic.add_midpoint(point)
        self.adjust_motor()

    def adjust_motor(self):
        m_x, m_y, m_z = self.motor.get_pos()
        e_x, e_y, e_z = self.elastic.get_second_point()
        yaw = np.arctan2(m_x-e_x, m_z-e_z)
        self.motor.change_orientation(0, yaw, 0)
        self.elastic.update_start()
        self.elastic.change_natural_length(self.elastic.get_unextended_length())

    def get_force(self):
        return self.elastic.get_force()

    def get_insertion2_loc(self):
        return self.elastic.get_insertion2_loc()

    def update(self, dt):
        self.elastic.update()

    def do_batch(self):
        self.motor.do_batch()

    def render(self):
        self.motor.render()
        self.elastic.render()


def get_tex(file):
    tex = pyglet.image.load(file).get_texture()
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    return pyglet.graphics.TextureGroup(tex)