import numpy as np
import math

def mul_matrix_mul(matrix_list): # multiplies a list of arrays
    I = np.identity(matrix_list[0].shape[0])
    new_matrix = I
    for matrix in matrix_list:
        new_matrix = np.matmul(new_matrix, matrix)
    return new_matrix


def get_quaternion_distance(point1, point2):
    return 1-np.dot(point1, point2)**2

def get_quaternion_angle(q1, q2):
    return math.acos(2*np.dot(q1,q2)**2-1)