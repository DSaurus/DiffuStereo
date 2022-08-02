import numpy as np
import math

def cross_3d(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], b[0]*a[2]-a[0]*b[2], a[0]*b[1]-b[0]*a[1]])


def rotationX(angle):
    return [
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)],
    ]


def rotationY(angle):
    return [
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)],
    ]


def rotationZ(angle):
    return [
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ]