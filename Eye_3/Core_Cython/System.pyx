include "cy_utils.pyx"
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class System:
    #cdef:
    #    cdef Eye eye
    #    cdef Elastic superior_rectus, inferior_rectus, medial_rectus, lateral_rectus, superior_oblique, inferior_oblique
    #    cdef Motor motor1, motor2, motor3

    def __cinit__(self):
        cdef:
            double sr_lat1, sr_long1, sr_lat2, sr_long2
            double ir_lat1, ir_long1, ir_lat2, ir_long2
            double lr_lat1, lr_long1, lr_lat2, lr_long2
            double mr_lat1, mr_long1, mr_lat2, mr_long2
            double so_lat1, so_long1, so_lat2, so_long2
            double io_lat1, io_long1, io_lat2, io_long2 


        # Objects
        self.eye = Eye()
        self.eye.change_pos(0, 0, 0)
        self.eye.change_scale(0.0753076410465764)

        # Superior and Inferior Rectus
        self.motor1 = Motor()
        self.motor1.change_scale(0.02, 0.1, 0.01)
        self.motor1.change_pos(0.436, 0, -0.0835)

        sr_long1 = 0
        sr_lat1 = 90
        sr_long2 = 82.47898517199053
        sr_lat2 = 44.75247673069133
        self.superior_rectus = Elastic(self.motor1, sr_long1, sr_lat1, self.eye, sr_long2, sr_lat2)
        self.superior_rectus.add_midpoint(0.2055, 0.0525, 0)

        ir_long1 = 0
        ir_lat1 = -90
        ir_long2 = 82.47898517199053
        ir_lat2 = -44.75247673069133
        self.inferior_rectus = Elastic(self.motor1, ir_long1, ir_lat1, self.eye, ir_long2, ir_lat2)
        self.inferior_rectus.add_midpoint(0.2055, -0.0525, 0)

        # Medial and Lateral Rectus
        self.motor2 = Motor()
        self.motor2.change_scale(0.02, 0.01, 0.14)
        self.motor2.change_pos(0.323, 0, 0)

        lr_long1 = -90
        lr_lat1 = 0
        lr_long2 = -84.66642703123067
        lr_lat2 = 0
        self.lateral_rectus = Elastic(self.motor2, lr_long1, lr_lat1, self.eye, lr_long2, lr_lat2)
        self.lateral_rectus.change_natural_length(0.24)
        # self.lateral_rectus.add_midpoint(np.array([2.5, 0, -1]))

        mr_long1 = 90
        mr_lat1 = 0
        mr_long2 = 84.66642703123067
        mr_lat2 = 0
        self.medial_rectus = Elastic(self.motor2, mr_long1, mr_lat1, self.eye, mr_long2, mr_lat2)
        self.medial_rectus.change_natural_length(0.24)
        # self.medial_rectus.add_midpoint(np.array([2.5, 0, 1]))

        # Superior and Inferior Oblique
        self.motor3 = Motor()
        self.motor3.change_scale(0.02, .1, 0.01)
        self.motor3.change_pos(0.436, 0, 0.0835)

        so_long1 = 0
        so_lat1 = 90
        so_long2 = -82.47898517199053
        so_lat2 = 44.75247673069133
        self.superior_oblique = Elastic(self.motor3, so_long1, so_lat1, self.eye, so_long2, so_lat2)
        self.superior_oblique.add_midpoint(0.2055, 0.0525, 0)

        io_long1 = 0
        io_lat1 = -90
        io_long2 = -82.47898517199053
        io_lat2 = -44.75247673069133
        self.inferior_oblique = Elastic(self.motor3, io_long1, io_lat1, self.eye, io_long2, io_lat2)
        self.inferior_oblique.add_midpoint(0.2055, -0.0525, 0)

        self.eye.add_parent(self.superior_rectus)
        self.eye.add_parent(self.inferior_rectus)
        self.eye.add_parent(self.medial_rectus)
        self.eye.add_parent(self.lateral_rectus)
        self.eye.add_parent(self.superior_oblique)
        self.eye.add_parent(self.inferior_oblique)
        
    cpdef void feed_input(self, double step1, double step2, double step3):  # TODO rename this?
        # TODO maybe treat step beforehand
        # step is assumed in radians
        self.motor1.change_orientation(0, 0, step1)
        self.motor2.change_orientation(0, step2, 0)
        self.motor3.change_orientation(0, 0, step3)

    cpdef (double, double, double) get_motor_position(self):
        cdef double motor1_orientation, motor2_orientation, motor3_orientation
        _, _, motor1_orientation = self.motor1.get_orientation()
        _, motor2_orientation, _ = self.motor2.get_orientation()
        _, _, motor3_orientation = self.motor3.get_orientation()
        return motor1_orientation, motor2_orientation, motor3_orientation

    cpdef (double, double, double, double) get_output(self):
        cdef double q0, qx, qy, qz
        q0, qx, qy, qz = self.eye.get_orientation()
        return q0, qx, qy, qz

    cpdef (double, double, double) get_eye_angular_velocity(self):
        return self.eye.get_angular_velocity()

    cdef void do_premovement(self, double time, double time_step, double action1, double action2, double action3):
        self.feed_input(action1, action2, action3)
        for _ in range(int(time/time_step)):
            self.update(time_step)

    cpdef void update(self, double dt):
        self.motor1.update(dt)
        self.motor2.update(dt)
        self.motor3.update(dt)
        self.superior_rectus.update(dt)
        self.inferior_rectus.update(dt)
        self.medial_rectus.update(dt)
        self.lateral_rectus.update(dt)
        self.superior_oblique.update(dt)
        self.inferior_oblique.update(dt)
        self.eye.update(dt)

    cpdef void reset(self):
        self.motor1.change_orientation(0, 0, 0)
        self.motor2.change_orientation(0, 0, 0)
        self.motor3.change_orientation(0, 0, 0)
        self.eye.change_orientation(0,0,0)
        self.eye.change_angular_velocity(0, 0, 0)
        self.superior_rectus.update(0)
        self.inferior_rectus.update(0)
        self.medial_rectus.update(0)
        self.lateral_rectus.update(0)
        self.superior_oblique.update(0)
        self.inferior_oblique.update(0)

    cpdef void change_eye_quaternion(self, double new_q0, double new_qx, double new_qy, double new_qz):
        self.eye.change_orientation_as_quat(new_q0, new_qx, new_qy, new_qz)
    
    cpdef void stop(self):
        self.eye.change_angular_velocity(0,0,0)

    cpdef get_objects(self):
        return self.motor1, self.motor2, self.motor3, self.superior_rectus, self.inferior_rectus, self.medial_rectus, self.lateral_rectus, self.superior_oblique, self.inferior_oblique, self.eye 


cdef class Elastic:
    #cdef:
    #    double k, natural_length
    #    double latitude1, longitude1, latitude2, longitude2
    #    double start_x, start_y, start_z, end_x, end_y, end_z
    #    double *midpoints
    #    int n_midpoints
    #    Motor obj1
    #    Eye obj2
    def __cinit__(self, Motor obj1, double long1, double lat1, Eye obj2, double long2, double lat2):
        self.k = -6.0
        self.natural_length = 0.36

        self.obj1 = obj1
        self.latitude1 = lat1
        self.longitude1 = long1
        self.obj2 = obj2
        self.latitude2 = lat2
        self.longitude2 = long2

        # self.start_x = self.start_y = self.start_z = None
        # self.end_x = self.end_y = self.end_z = None
        
        self.midpoints
        self.n_midpoints = 0

        self.update(0)

    cdef void add_midpoint(self, double point_x, double point_y, double point_z):
        cdef int aux_n_midpoints = 3*self.n_midpoints+3
        self.midpoints = <double*>PyMem_Realloc(self.midpoints, (aux_n_midpoints)*sizeof(double))
        self.n_midpoints += 1
        self.midpoints[aux_n_midpoints-3] = point_x
        self.midpoints[aux_n_midpoints-2] = point_y        
        self.midpoints[aux_n_midpoints-1] = point_z

    cpdef (double, double) get_insertion2_loc(self):  # TODO change this (quite hateful)
        return self.longitude2, self.latitude2

    cdef (double, double, double) get_force_direction(self):
        cdef:
            double prev_x, prev_y, prev_z
            double vector_x, vector_y, vector_z, mag
            int aux_n_midpoints = 3*self.n_midpoints

        if self.n_midpoints == 0:
            prev_x = self.start_x
            prev_y = self.start_y
            prev_z = self.start_z
        else:
            prev_x = self.midpoints[aux_n_midpoints-3]
            prev_y = self.midpoints[aux_n_midpoints-2]
            prev_z = self.midpoints[aux_n_midpoints-1]

        vector_x = self.end_x-prev_x
        vector_y = self.end_y-prev_y
        vector_z = self.end_z-prev_z

        mag = sqrt(vector_x**2+vector_y**2+vector_z**2)

        return vector_x/mag, vector_y/mag, vector_z/mag

    cdef double get_force_magnitude(self):
        return self.k / self.natural_length * max(self.get_total_length() - self.natural_length, 0)

    cdef double get_total_length(self):
        cdef:
            double total_length = 0
            double prev_x, prev_y, prev_z
            double len_x, len_y, len_z
            int i, aux_i
        prev_x = self.start_x
        prev_y = self.start_y
        prev_z = self.start_z
        for i in range(self.n_midpoints):
            aux_i = i*3
            len_x = self.midpoints[aux_i]-prev_x
            len_y = self.midpoints[aux_i+1]-prev_y
            len_z = self.midpoints[aux_i+2]-prev_z
            total_length += sqrt(len_x**2+len_y**2+len_z**2)
            prev_x = self.midpoints[aux_i]
            prev_y = self.midpoints[aux_i+1]
            prev_z = self.midpoints[aux_i+2]
        total_length += sqrt((self.end_x-prev_x)**2+(self.end_y-prev_y)**2+(self.end_z-prev_z)**2)
        return total_length

    cpdef (double, double, double) get_force(self):
        cdef double f, dir_x, dir_y,dir_z
        f = self.get_force_magnitude()
        dir_x, dir_y, dir_z = self.get_force_direction()
        return f*dir_x, f*dir_y, f*dir_z

    cdef void change_natural_length(self, double length):
        self.natural_length = length

    cdef void update(self, double dt):
        cdef:
            double center_x, center_y, center_z, insertion_x, insertion_y, insertion_z

        center_x, center_y, center_z = self.obj1.get_pos()
        insertion_x, insertion_y, insertion_z = self.obj1.get_point(self.longitude1, self.latitude1)

        self.start_x = center_x+insertion_x
        self.start_y = center_y+insertion_y
        self.start_z = center_z+insertion_z

        center_x, center_y, center_z = self.obj2.get_pos()
        insertion_x, insertion_y, insertion_z = self.obj2.get_point(self.longitude2, self.latitude2)

        self.end_x = center_x+insertion_x
        self.end_y = center_y+insertion_y
        self.end_z = center_z+insertion_z

    cpdef list get_midpoints(self):
        aux = []
        for i in range(self.n_midpoints*3):
            aux.append(self.midpoints[i])
        return aux
    def __dealloc__(self):
        if self.midpoints:
            PyMem_Free(self.midpoints)    

cdef class Motor:
    #cdef:
    #    double pos_x, pos_y, pos_z
    #    double roll, yaw, pitch
    #    double scale_x, scale_y, scale_z

    def __cinit__(self):
        self.pos_x = self.pos_y = self.pos_z = 0
        self.roll = self.yaw = self.pitch = 0
        self.scale_x = self.scale_y = self.scale_z = 1

    cdef (double, double, double) get_pos(self):
        return self.pos_x, self.pos_y, self.pos_z

    cdef (double, double, double) get_orientation(self):
        return self.roll, self.yaw, self.pitch

    cdef (double, double, double) get_point(self, double longitude, double latitude):
        cdef:
            double point_x, point_y, point_z
            double q0, qx, qy, qz
        latitude = deg2rad(latitude)
        longitude = deg2rad(longitude)
        point_x = self.scale_x * -cos(latitude)*cos(longitude)
        point_y = self.scale_y * sin(latitude)
        point_z = self.scale_z * cos(latitude)*sin(longitude)

        q0, qx, qy, qz = rotation_vector_to_quaternion(self.roll, self.yaw, self.pitch)

        return quaternion_rotation(q0, qx, qy, qz, point_x, point_y, point_z)

    cdef void change_pos(self, double pos_x, double pos_y, double pos_z):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z

    cdef void change_orientation(self, double roll, double yaw, double pitch):
        self.roll = roll
        self.yaw = yaw
        self.pitch = pitch
    
    cdef change_scale(self, double scale_x, double scale_y, double scale_z):
        self.scale_x = scale_x * (scale_x != 0) + (scale_x == 0)
        self.scale_y = scale_y * (scale_y != 0) + (scale_y == 0)
        self.scale_z = scale_z * (scale_z != 0) + (scale_z == 0)

    cdef void update(self, double dt):
        pass


cdef class Eye:
    #cdef:
    #    double friction_coefficient, static_friction_torque
    #    double pos_x, pos_y, pos_z  # position
    #    double q0, qx, qy, qz  # quaternion orientation
    #    double w_x, w_y, w_z  # angular velocity
    #    double scale_x, scale_y, scale_z
    #    double* inertia_tensor
    #    double center_mass_x, center_mass_y, center_mass_z
    #    double mass
    #    double gravity_x, gravity_y, gravity_z
    #    list parent_list 
    def __cinit__(self):
        # Parameters
        self.friction_coefficient = -0.02
        self.static_friction_torque = 0.006

        # state
        self.pos_x = self.pos_y = self.pos_z = 0
        self.q0 = 1
        self.qx = self.qy = self.qz = 0
        self.w_x = self.w_y = self.w_z = 0
        self.scale_x = self.scale_y = scale_z = 1
        self.inertia_tensor = <double*>PyMem_Malloc(9*sizeof(double))
        for i, val in enumerate([0.00004759, 0, 0, 0, 0.00003956, 0, 0, 0, 0.00004316]):
            self.inertia_tensor[i] = val
        self.center_mass_x = -0.02
        self.center_mass_y = self.center_mass_z = 0
        self.mass = 0.248
        self.gravity_x = self.gravity_z = 0
        self.gravity_y = -9.81

        self.parent_list = []

    cdef (double, double, double) get_pos(self):
        return self.pos_x, self.pos_y, self.pos_z

    cdef (double, double, double, double) get_orientation(self):
        return self.q0, self.qx, self.qy, self.qz

    cdef (double, double, double) get_angular_velocity(self):
        return self.w_x, self.w_y, self.w_z

    cdef (double, double, double) get_point(self, double longitude, double latitude):
        cdef double point_x, point_y, point_z

        latitude = deg2rad(latitude)
        longitude = deg2rad(longitude)
        
        point_x = self.scale_x*(-cos(latitude)) * cos(longitude)
        point_y = self.scale_y*sin(latitude)
        point_z = self.scale_z*cos(latitude)*sin(longitude)

        return quaternion_rotation(self.q0, self.qx, self.qy, self.qz, point_x, point_y, point_z)

    cdef double* get_current_inertia_tensor(self, double* R):
        cdef double* Rt = transpose_matrix(R, 3, 3)
        cdef double* aux_matrix = multiply_matrices(R, 3, 3, self.inertia_tensor, 3, 3)
        cdef double* final_tensor = multiply_matrices(aux_matrix, 3, 3, Rt, 3, 3)
        if Rt:
            PyMem_Free(Rt)
        if aux_matrix:
            PyMem_Free(aux_matrix)
        return final_tensor

    cdef (double, double, double) get_gravity_torque(self, double* R):
        cdef double current_center_of_mass_x = R[0]*self.center_mass_x+R[1]*self.center_mass_y+R[2]*self.center_mass_z
        cdef double current_center_of_mass_y = R[3]*self.center_mass_x+R[4]*self.center_mass_y+R[5]*self.center_mass_z
        cdef double current_center_of_mass_z = R[6]*self.center_mass_x+R[7]*self.center_mass_y+R[8]*self.center_mass_z
        cdef double force_x = self.gravity_x*self.mass
        cdef double force_y = self.gravity_y*self.mass
        cdef double force_z = self.gravity_z*self.mass
        return cross_product(current_center_of_mass_x, current_center_of_mass_y, current_center_of_mass_z, force_x, force_y, force_z)
        
    cdef void change_pos(self, double pos_x, double pos_y, double pos_z): # TODO change name from change to set
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z

    cdef void change_orientation(self, double roll, double yaw, double pitch):
        self.q0, self.qx, self.qy, self.qz = rotation_vector_to_quaternion(roll, yaw, pitch)

    cdef void change_orientation_as_quat(self, double q0, double qx, double qy, double qz):
        self.q0 = q0
        self.qx = qx
        self.qy = qy
        self.qz = qz

    cdef void change_scale(self, double scale):
        if scale == 0:
            return

        self.scale_x = self.scale_y = self.scale_z = scale

    cdef void change_angular_velocity(self, double w_x, double w_y, double w_z):
        self.w_x = w_x
        self.w_y = w_y
        self.w_z = w_z

    cpdef void add_parent(self, Elastic parent):
        self.parent_list.append(parent)

    cdef void update(self, double dt):  # TODO make sure dt is taken from somewhere
        cdef:
            double total_torque_x, total_torque_y, total_torque_z
            double torque_friction_x, torque_friction_y, torque_friction_z
            double torque_gravity_x, torque_gravity_y, torque_gravity_z
            double torque_elastic_x, torque_elastic_y, torque_elastic_z
            double this_torque_x, this_torque_y, this_torque_z
            double long2, lat2, ix, iy, iz, fx, fy, fz
            double angular_acceleration_x, angular_acceleration_y, angular_acceleration_z
            bint is_moving
            double *R # rotation_matrix
            double *I # inertia_tensor
            double *inv_I, # inversed inertia_tensor
            double dot_q0, dot_qx, dot_qy, dot_qz, q_mag

        

        torque_gravity_x = torque_gravity_y = torque_gravity_z = 0
        torque_elastic_x = torque_elastic_y = torque_elastic_z = 0

        
        for parent in self.parent_list:
            long2, lat2 = parent.get_insertion2_loc()
            ix, iy, iz = self.get_point(long2, lat2)
            ix -= self.pos_x
            iy -= self.pos_y
            iz -= self.pos_z

            fx, fy, fz = parent.get_force()
            this_torque_x, this_torque_y, this_torque_z = cross_product(ix, iy, iz, fx, fy, fz)
            torque_elastic_x += this_torque_x
            torque_elastic_y += this_torque_y
            torque_elastic_z += this_torque_z

        
        torque_friction_x = self.friction_coefficient*self.w_x
        torque_friction_y = self.friction_coefficient*self.w_y
        torque_friction_z = self.friction_coefficient*self.w_z

        R = quaternion_to_rotation_matrix(self.q0, self.qx, self.qy, self.qz)
        # torque_gravity_x, torque_gravity_y, torque_gravity_z = self.get_gravity_torque(R)
        
        total_torque_x = torque_elastic_x + torque_friction_x + torque_gravity_x
        total_torque_y = torque_elastic_y + torque_friction_y + torque_gravity_y
        total_torque_z = torque_elastic_z + torque_friction_z + torque_gravity_z 
        
        is_moving = ((total_torque_x**2+total_torque_y**2+total_torque_z**2)>self.static_friction_torque**2 or (self.w_x**2+self.w_y**2+self.w_z**2) > 0.00001**2)
        
        total_torque_x = total_torque_x * is_moving
        total_torque_y = total_torque_y * is_moving
        total_torque_z = total_torque_z * is_moving

        I = self.get_current_inertia_tensor(R)
        
        inv_I = inverse_matrix3(I, 3, 3)
        
        angular_acceleration_x = inv_I[0]*total_torque_x+inv_I[1]*total_torque_y+inv_I[2]*total_torque_z
        angular_acceleration_y = inv_I[3]*total_torque_x+inv_I[4]*total_torque_y+inv_I[5]*total_torque_z
        angular_acceleration_z = inv_I[6]*total_torque_x+inv_I[7]*total_torque_y+inv_I[8]*total_torque_z
        
        self.w_x = (self.w_x+angular_acceleration_x*dt)*is_moving
        self.w_y = (self.w_y+angular_acceleration_y*dt)*is_moving
        self.w_z = (self.w_z+angular_acceleration_z*dt)*is_moving

        dot_q0, dot_qx, dot_qy, dot_qz = quaternion_multiplication(0, self.w_x, self.w_y, self.w_z, self.q0, self.qx, self.qy, self.qz)
        
        self.q0 += 0.5*dot_q0*dt
        self.qx += 0.5*dot_qx*dt
        self.qy += 0.5*dot_qy*dt
        self.qz += 0.5*dot_qz*dt
        
        q_mag = sqrt(self.q0**2+self.qx**2+self.qy**2+self.qz**2)
        self.q0 /= q_mag
        self.qx /= q_mag
        self.qy /= q_mag
        self.qz /= q_mag
        if R:
            PyMem_Free(R)
        if I:
            PyMem_Free(I)
        if inv_I:
            PyMem_Free(inv_I)

    def __dealloc__(self):
        if self.inertia_tensor:
            PyMem_Free(self.inertia_tensor)