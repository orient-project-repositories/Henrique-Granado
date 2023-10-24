cdef class Motor:
    cdef:
        double pos_x, pos_y, pos_z
        double roll, yaw, pitch
        double scale_x, scale_y, scale_z
    cdef (double, double, double) get_pos(self)
    cdef (double, double, double) get_orientation(self)
    cdef (double, double, double) get_point(self, double longitude, double latitude)
    cdef void change_pos(self, double pos_x, double pos_y, double pos_z)
    cdef void change_orientation(self, double roll, double yaw, double pitch)
    cdef change_scale(self, double scale_x, double scale_y, double scale_z)
    cdef void update(self, double dt)

cdef class Eye:
    cdef:
        double friction_coefficient, static_friction_torque
        double pos_x, pos_y, pos_z  # position
        double q0, qx, qy, qz  # quaternion orientation
        double w_x, w_y, w_z  # angular velocity
        double scale_x, scale_y, scale_z
        double* inertia_tensor
        double center_mass_x, center_mass_y, center_mass_z
        double mass
        double gravity_x, gravity_y, gravity_z
        list parent_list
    cdef (double, double, double) get_pos(self)
    cdef (double, double, double, double) get_orientation(self)
    cdef (double, double, double) get_angular_velocity(self)
    cdef (double, double, double) get_point(self, double longitude, double latitude)
    cdef double* get_current_inertia_tensor(self, double* R)
    cdef (double, double, double) get_gravity_torque(self, double* R)
    cdef void change_pos(self, double pos_x, double pos_y, double pos_z)
    cdef void change_orientation(self, double roll, double yaw, double pitch)
    cdef void change_orientation_as_quat(self, double q0, double qx, double qy, double qz)
    cdef void change_scale(self, double scale)
    cdef void change_angular_velocity(self, double w_x, double w_y, double w_z)
    cpdef void add_parent(self, Elastic parent)
    cdef void update(self, double dt)

cdef class Elastic:
    cdef:
        double k, natural_length
        double latitude1, longitude1, latitude2, longitude2
        double start_x, start_y, start_z, end_x, end_y, end_z
        double *midpoints
        int n_midpoints
        Motor obj1
        Eye obj2
    cdef void add_midpoint(self, double point_x, double point_y, double point_z)
    cpdef (double, double) get_insertion2_loc(self)
    cdef (double, double, double) get_force_direction(self)
    cdef double get_force_magnitude(self)  # TODO make it just cdef
    cdef double get_total_length(self)  # TODO make it just cdef
    cpdef (double, double, double) get_force(self)
    cdef void change_natural_length(self, double length)
    cpdef list get_midpoints(self)
    cdef void update(self, double dt)

cdef class System:
    cdef:
        Eye eye
        Elastic superior_rectus, inferior_rectus, medial_rectus, lateral_rectus, superior_oblique, inferior_oblique
        Motor motor1, motor2, motor3
    cpdef void feed_input(self, double step1, double step2, double step3)
    cpdef (double, double, double) get_motor_position(self)
    cpdef (double, double, double, double) get_output(self)
    cpdef (double, double, double) get_eye_angular_velocity(self)
    cdef void do_premovement(self, double time, double time_step, double action1, double action2, double action3)
    cpdef void update(self, double dt)
    cpdef void reset(self)
    cpdef void change_eye_quaternion(self, double new_q0, double new_qx, double new_qy, double new_qz)
    cpdef void stop(self)
    cpdef get_objects(self)