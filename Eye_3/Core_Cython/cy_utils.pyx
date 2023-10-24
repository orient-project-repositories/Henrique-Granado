from cpython.mem cimport PyMem_Malloc
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI
import numpy as np
# basic math

cdef double deg2rad(double deg):
	return deg*M_PI/180.0

cdef double rad2deg(double rad):
	return rad*180.0/M_PI


# basic vector/matrix operations
cdef (double, double, double) cross_product(double ux, double uy, double uz, double vx, double vy, double vz):
	cdef double rx = uy*vz-uz*vy
	cdef double ry = uz*vx-ux*vz
	cdef double rz = ux*vy-uy*vx
	return rx, ry, rz


cdef double* multiply_matrices(double* A, int row_A, int col_A, double* B, int row_B, int col_B):
	cdef int N = row_A*col_B
	cdef double* C = <double*>PyMem_Malloc(N*sizeof(double))
	cdef int i, j, k
	cdef int n = col_B
	# Initialize C
	for i in range(N):
		C[i] = 0

	# Multiply
	for i in range(row_A):
		for j in range(col_B):
			for k in range(col_A):
				C[i*n+j] += A[i*col_A+k]*B[k*n+j]

	return C


cdef double* transpose_matrix(double* M, int rows, int cols):
	cdef double* T = <double*>PyMem_Malloc(rows*cols*sizeof(double))
	cdef int i, j
	for i in range(rows):
		for j in range(cols):
			T[i*cols+j] = M[i+j*rows]
	return T


cdef double* inverse_matrix3(double* M, int rows, int cols):  # TODO finish this
	if rows != cols != 3:
		raise NotImplementedError("Rows and Columns don't match or neither has size 3!")
	cdef double det = determinant3(M, rows, cols)
	if det == 0:
		raise ValueError("Determinant is 0: Singular Matrix!")

	cdef double* inv = adjoint3(M, rows, cols)
	cdef int i
	for i in range(rows*cols):
		inv[i] /= det
	return inv


cdef double determinant3(double* M, int rows, int cols):
	if rows != cols != 3:
		raise ValueError("Rows and Columns don't match or neither has size 3!")
	return M[0]*(M[4]*M[8]-M[5]*M[7])+M[1]*(M[5]*M[6]-M[3]*M[8])+M[2]*(M[3]*M[7]-M[4]*M[6])


cdef double determinant2(double* M, int rows, int cols):
	if rows != cols != 2:
		raise ValueError("Rows and Columns don't match or neither has size 3!")
	return M[0]*M[3]-M[1]*M[2]


cdef double* adjoint3(double* M, int rows, int cols):
	if rows != cols != 3:
		raise ValueError("Rows and Columns don't match or neither has size 3!")
	cdef double* adj = <double*>PyMem_Malloc(rows*cols*sizeof(double))
	adj[0] = M[4]*M[8]-M[5]*M[7]
	adj[1] = M[2]*M[7]-M[1]*M[8]
	adj[2] = M[1]*M[5]-M[2]*M[4]
	adj[3] = M[5]*M[6]-M[3]*M[8]
	adj[4] = M[0]*M[8]-M[2]*M[6]
	adj[5] = M[2]*M[3]-M[0]*M[5]
	adj[6] = M[3]*M[7]-M[4]*M[6]
	adj[7] = M[1]*M[6]-M[0]*M[7]
	adj[8] = M[0]*M[4]-M[1]*M[3]
	return adj


# quaternion operations and convertions
cdef double* quaternion_to_rotation_matrix(double q0, double qx, double qy, double qz):
	cdef double* R = <double*>PyMem_Malloc(9*sizeof(double))
	R[0] = 1-2*(qy**2+qz**2)
	R[1] = 2*(qx*qy-qz*q0)
	R[2] = 2*(qx*qz+qy*q0)
	R[3] = 2*(qx*qy+qz*q0)
	R[4] = 1-2*(qx**2+qz**2)
	R[5] = 2*(qy*qz-qx*q0)
	R[6] = 2*(qx*qz-qy*q0)
	R[7] = 2*(qy*qz+qx*q0)
	R[8] = 1-2*(qx**2+qy**2)
	return R


cdef (double, double, double, double) rotation_vector_to_quaternion(double roll, double yaw, double pitch):
	if roll == yaw == pitch == 0:
		return 1.0, 0.0, 0.0, 0.0
	cdef double theta = sqrt(roll**2+yaw**2+pitch**2)
	cdef double q0 = cos(theta/2)
	cdef double S = sin(theta/2)

	cdef double qx = S*roll/theta
	cdef double qy = S*yaw/theta
	cdef double qz = S*pitch/theta

	return q0, qx, qy, qz


cdef (double, double, double, double) quaternion_multiplication(double q0, double qx, double qy, double qz, double p0, double px, double py, double pz):
	cdef double r0 = q0*p0-qx*px-qy*py-qz*pz
	cdef double rx = qx*p0+q0*px+qy*pz-qz*py
	cdef double ry = q0*py-qx*pz+qy*p0+qz*px
	cdef double rz = q0*pz+qx*py-qy*px+qz*p0
	return r0, rx, ry, rz


cdef (double, double, double) quaternion_rotation(double q0, double qx, double qy, double qz, double point_x, double point_y, double point_z):
	cdef double aux_0, aux_x, aux_y, aux_z, res_x, res_y, res_z
	aux_0, aux_x, aux_y, aux_z = quaternion_multiplication(q0, qx, qy, qz, 0, point_x, point_y, point_z)
	_, res_x, res_y, res_z = quaternion_multiplication(aux_0, aux_x, aux_y, aux_z, q0, -qx, -qy, -qz)
	return res_x, res_y, res_z


# cdef class Test:
# 	def test_deg2rad(self):
# 		v = np.random.uniform(-360, 360)
# 		print(deg2rad(v), np.radians(v))
#	def test_rad2deg(self):
#		v = np.random.uniform(-2*np.pi, 2*np.pi)
#		print(rad2deg(v), np.degrees(v))
#	def test_cross_product(self):
#		u = np.random.random(3)
#		v = np.random.random(3)
#		c_fun = np.array(cross_product(u[0], u[1], u[2], v[0], v[1], v[2]))
#		np_fun = np.cross(u,v)
#		print(c_fun, np_fun)