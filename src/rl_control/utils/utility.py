import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from math import radians, sin, cos

from six.moves import cPickle as pickle 
from scipy.special import comb


def plot(name, length):
    data = pd.read_csv(name).to_numpy()
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 3, 1)
    plt.plot(data[:length, 0])
    plt.title('torque_knee'), plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(data[:length, 1], label='q_knee_desired')
    plt.plot(data[:length, 2], label='q_knee_actual')
    plt.title('q_knee'), plt.legend(), plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(data[:length, 3])
    plt.title('torque_foot'), plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(data[:length, 4], label='foot_x')
    plt.plot(data[:length, 5], label='base_x')
    plt.title('toe'), plt.legend(), plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(data[:length, 6])
    plt.title('shin_q5'), plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(data[:length, 7])
    plt.title('tarsus_q6'), plt.grid()
    plt.show()


def plot_crank(name, length):
    data = pd.read_csv(name, usecols=['foot_crank']).to_numpy()
    plt.figure(figsize=(18, 10))
    # plt.subplot(2, 3, 1)
    plt.plot(data[:length, 0])
    plt.title('left_foot_crank'), plt.grid()
    plt.show()


# function convert array to list
def get_list(array, dim):
    output = []
    for i in range(dim):
        output.append(array[i])
    return output


# q1 * q2 =
# (w1*w2 - x1*x2 - y1*y2 - z1*z2) +
# (w1*x2 + x1*w2 + y1*z2 - z1*y2) i +
# (w1*y2 - x1*z2 + y1*w2 + z1*x2) j +
# (w1*z2 + x1*y2 - y1*x2 + z1*w2) k
# function return difference between q1 and q2
def QuatDiff(q1, q2):
    # q = [w, x, y, z]
    for i in range(1, 4):
        q1[i] = -q1[i]
    # dq = q1.conjugate() * q2
    q_diff = [q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
              q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
              q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
              q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]]
    return q_diff


# function return the theta of dq = [w, x, y, z]
def QuatTheta(dq):
    # normalize dq = dq / (dq_w^2 + dq_x^2 + dq_y^2 + dq_z^2 +)
    if dq[0] > 1:
        dq = dq / (np.sum(np.asarray(dq) * np.asarray(dq)))
    # get sin_theta
    sin_theta = math.sqrt(1 - dq[0] * dq[0])
    # ignore sin_theta around 0
    if sin_theta > 0.0001:
        angle = (2 * math.acos(dq[0]) * 180) / math.pi
        angle = angle % 360
        if angle < 0:
            angle = angle + 360
    else:
        theta = 0
        return theta
    # get theta from angle
    theta = angle * math.pi / 180
    assert 0 <= theta <= 2*np.pi
    return theta


# function return the difference between q1 and q2
def QuatDiffTheta(q1, q2):
    dq = QuatDiff(q1, q2)
    theta = QuatTheta(dq)
    # theta in range (0, 2pi)
    return theta


# roll, pitch, yaw: rx, ry, rz
def ThetaQuat(rxryrz):
    rx = rxryrz[0]
    ry = rxryrz[1]
    rz = rxryrz[2]
    q = np.vstack([np.cos(rx / 2) * np.cos(ry / 2) * np.cos(rz / 2) + np.sin(rx / 2) * np.sin(ry / 2) * np.sin(rz / 2),
                   np.sin(rx / 2) * np.cos(ry / 2) * np.cos(rz / 2) - np.cos(rx / 2) * np.sin(ry / 2) * np.sin(rz / 2),
                   np.cos(rx / 2) * np.sin(ry / 2) * np.cos(rz / 2) + np.sin(rx / 2) * np.cos(ry / 2) * np.sin(rz / 2),
                   np.cos(rx / 2) * np.cos(ry / 2) * np.sin(rz / 2) - np.sin(rx / 2) * np.sin(ry / 2) * np.cos(rz / 2)])
    return q.ravel()

def Quat2Rxyz(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    # if sinp<-1 or sinp >1:
    #     i=1
    sinp = np.clip(sinp, -1, 1)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def dbezier(coeff, s):
    dcoeff = __diff_coeff(coeff)
    fcn = bezier(dcoeff,s)
    return fcn

def __binomial(i, n):
    """Binomial coefficient"""
    return math.factorial(n) / float(
        math.factorial(i) * math.factorial(n - i))

def __bernstein(t, i, n):
    """Bernstein polynom"""
    return __binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))

def bezier(coeff, s):
    """Calculate coordinate of a point in the bezier curve"""
    n, m = coeff.shape[0], coeff.shape[1]
    m = m - 1
    fcn = np.zeros((n, 1))
    for k in range(m+1):
        fcn += coeff[:,k].reshape((n,1)) * __bernstein(s, k, m)
    return fcn.reshape((n,))

def __diff_coeff(coeff):
    M = coeff.shape[1] - 1
    A = np.zeros((M, M+1))

    for i in range(M):
        A[i,i] = -(M-i)*comb(M,i)/comb(M-1, i)
        A[i,i+1] = (i+1)*comb(M,i+1)/comb(M-1,i)
    
    A[M-1,M] = M*comb(M,M)
    dcoeff = coeff@(A.T)
    return dcoeff

def first_order_filter(prev, new, para):
    return prev*(1-para)+new*para

# yaw pitch roll (radians) is the rotation of base related to global frame
def __global_to_local_rotation_matix(yaw, pitch, roll):
    Rgl = np.identity(3) 
    yawed_by = np.identity(3)
    yawed_by[0, 0] = cos(yaw)
    yawed_by[0, 1] = -(sin(yaw))
    yawed_by[1, 0] = sin(yaw)
    yawed_by[1, 1] = cos(yaw)
    Rgl = np.dot(Rgl, yawed_by)
    pitched_by = np.identity(3)
    pitched_by[0, 0] = cos(pitch)
    pitched_by[0, 2] = sin(pitch)
    pitched_by[2, 0] = -(sin(pitch))
    pitched_by[2, 2] = cos(pitch)
    Rgl = np.dot(Rgl, pitched_by)
    rolled_by = np.identity(3)
    rolled_by[1, 1] = cos(roll)
    rolled_by[1, 2] = -(sin(roll))
    rolled_by[2, 1] = sin(roll)
    rolled_by[2, 2] = cos(roll)
    Rgl = np.dot(Rgl, rolled_by)
    return Rgl

# l_a = Rab@l_b
def convert_vec_global_to_local(x_g, y_g, rot_rpy):
    roll, pitch, yaw = rot_rpy[0], rot_rpy[1], rot_rpy[2]
    v_g = np.array([[x_g, y_g, 0]]).T
    Rgl = __global_to_local_rotation_matix(yaw, pitch, roll)
    v_l = np.dot(Rgl.T, v_g).flatten()
    return v_l[0], v_l[1]

# l_a = Rab@l_b
def convert_vec_local_to_global(x_l, y_l, rot_rpy):
    roll, pitch, yaw = rot_rpy[0], rot_rpy[1], rot_rpy[2]
    v_l = np.array([[x_l, y_l, 0]]).T
    Rgl = __global_to_local_rotation_matix(yaw, pitch, roll)
    v_g = np.dot(Rgl, v_l).flatten()
    return v_g[0], v_g[1]

def test_transform():
    x_global, y_global = convert_vec_local_to_global(0,-1,[np.radians(90),0,0])
    print(x_global, y_global)
    x_local, y_local = convert_vec_global_to_local(0,-1,[np.radians(90),0,0])
    print(x_local, y_local)

#########################################
#              older version            #
#########################################
# def _local2global(self, x, y, rot_vec):
#     roll, pitch, yaw = rot_vec
#     xl, yl = x, y
#     # NOTE: pitch is first, no roll
#     # rotate pitch
#     xl = xl*np.cos(pitch) 
#     # rotate yaw
#     xg = xl*np.cos(yaw) - yl*np.sin(yaw) 
#     yg = xl*np.sin(yaw) + yl*np.cos(yaw)
#     return xg, yg

# # kinematics transform is hardcoded 
# # it can only do yaw+pitch transform in xy plane
# # TODO: to use a DH table if needed
def global2local(x, y, rot_vec):
    roll, pitch, yaw = rot_vec
    xg, yg = x, y
    # NOTE: yaw is first, no roll
    # rotate yaw
    xl = xg*np.cos(yaw) + yg*np.sin(yaw)
    yl = -xg*np.sin(yaw) + yg*np.cos(yaw)
    # rotate pitch
    xl = xl*np.cos(pitch) 
    return xl, yl

# def __test_transform(self):
#     # NOTE: output of y should be the same
#     xg, yg = 3, -2
#     rot_vec = np.array([0.0, np.radians(-10.0), np.radians(30.0)])
#     xl, yl = self._global2local(xg,yg,rot_vec)
#     xg_p, yg_p = self._local2global(xl,yl,rot_vec)
#     print(xg_p, yg_p)
#     xl, yl = 4, 3
#     rot_vec = np.array([0.0, np.radians(10.0), np.radians(-30.0)])
#     xg, yg = self._local2global(xl,yl,rot_vec)
#     xl_p, yl_p = self._global2local(xg,yg,rot_vec)
#     print(xl_p, yl_p)