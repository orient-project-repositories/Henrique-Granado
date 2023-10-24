# basic math
cdef double deg2rad(double deg)

cdef double rad2deg(double rad)


# basic vector/matrix operations
cdef (double, double, double) cross_product(double ux, double uy, double uz, double vx, double vy, double vz)

cdef double* multiply_matrices(double* A, int row_A, int col_A, double* B, int row_B, int col_B)

cdef double* transpose_matrix(double* M, int rows, int cols)

cdef double* inverse_matrix3(double* M, int rows, int cols)

cdef double determinant3(double* M, int rows, int cols)

cdef double determinant2(double* M, int rows, int cols)

cdef double* adjoint3(double* M, int rows, int cols)


# quaternion operations and convertions
cdef double* quaternion_to_rotation_matrix(double q0, double qx, double qy, double qz)

cdef (double, double, double, double) rotation_vector_to_quaternion(double roll, double yaw, double pitch)

cdef (double, double, double, double) quaternion_multiplication(double q0, double qx, double qy, double qz, double p0, double px, double py, double pz)

cdef (double, double, double) quaternion_rotation(double q0, double qx, double qy, double qz, double point_x, double point_y, double point_z)
