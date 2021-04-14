# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:22:38 2020

@author: Chuong Nguyen (ngu10t) <chuong.nguyen@csiro.au>
"""

'''
This script performs quantitave evaluation of the camera poses obtained by 
3D reconstruction softwares Agisoft of Photoscan and Meshroom of Alicevision 
from the real stacked images for both moving and fixed lens setups.
'''

import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as plt
plt.style.use('classic')
import cv2
from xml.dom import minidom
import xmltodict
import json
import math

'''
# # load cameras poses by Agisoft Photoscan
# infile = 'cameras_fixed_lens_agisoft_proposed.xml'
# infile = 'cameras_moving_lens_agisoft_proposed.xml'
# infile = 'cameras_fixed_lens_agisoft_helicon.xml'
# infile = 'cameras_moving_lens_agisoft_helicon.xml'

with open(infile, 'r') as xml_file:
     my_dict = xmltodict.parse(xml_file. read())
     cameras = my_dict['document']['chunk']['cameras']['camera']
     extrinsics = []
     for camera in cameras:
         if 'transform' in camera.keys():
             transform = [float(s) for s in camera['transform'].split(' ')]
             extrinsics.append(np.array(transform).reshape([4,4]))
         else:
             e = np.empty((4,4))
             e[:] = np.NaN
             extrinsics.append(e)

     if 'fixed' in infile:
         M = 1.7 # fixed lens
         #M = 0.50711 # fixed lens blender
     else:
         M = 1.3  # moving lens
         #M = 0.7647  # moving lens blender
'''

# # load cameras poses by Alicevision Meshroom
infile = 'fixed_lens_proposed_2nd_cameras.sfm'
#infile = 'moving_lens_propose_cameras.sfm'
#infile = 'fixed_lens_helicon_cameras.sfm'
#infile = 'moving_lens_helicon_cameras.sfm'

with open(infile, 'r') as json_file:
    my_dict = json.load(json_file)
    ids_paths = [(view['viewId'], view['path']) for view in my_dict['views']]
    ids_paths.sort(key=lambda tup: tup[1])  # sorts in place
    ids = [id for id, path in ids_paths]

    poses = my_dict['poses']
    extrinsics = [np.ones([3, 4])*np.nan]*len(ids)
    # for id, path in ids_paths:
    for pose in poses:
        rotation = [float(num) for num in pose['pose']['transform']['rotation']]
        center = [float(num) for num in pose['pose']['transform']['center']]
        rotation = np.array(rotation).reshape([3,3])
        center = np.array(center).reshape([3,1])
        mat = np.hstack([rotation, center])
        # extrinsics.append(mat)
        id = pose['poseId']
        extrinsics[ids.index(id)] = mat
    if 'fixed' in infile:
        M = 1.7 # fixed lens
        #M = 0.50711 # fixed lens blender
    else:
        M = 1.3  # moving lens
        #M = 0.7647  # moving lens blender


fL = 65 #mm
d0 = fL*(M+1)/M
d1 = fL*(M+1)
print('d0 [mm]:', d0)
print('d1 [mm]:', d1)

positions = np.array([mat[:3,3] for mat in extrinsics if not np.isnan(mat.sum())])

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2], 'o')

# compute rotation vector
tilt_pos = []
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
for i in range(5):
    pos = positions[i*24:(i+1)*24,:]
    centre = pos.mean(axis=0)
    tilt_pos.append(pos - centre)
    ax.scatter(tilt_pos[i][:,0], tilt_pos[i][:,1], tilt_pos[i][:,2], 'o')
positions2 = np.vstack(tilt_pos)


# find x in Ax = B
A = np.ones_like(positions2)
A[:,:2] = positions2[:,:2]
B = -positions2[:,2]
x = np.linalg.inv(A.T @ A) @ A.T @ B
n = x.copy()
n[2] = 1

no_iters = 5
positions4 = positions2
for i in range(no_iters):
    # find distance of positions to fitted planes for outlier removal
    # parameters of plane equation ax+by+cz=d
    a, b, c, d = x[0], x[1], 1, -x[2]
    positions3 = np.append(positions2, np.ones([positions2.shape[0],1]), axis=1)
    distances = np.abs(positions3 @ np.array([a, b, c, d]).T) / np.linalg.norm(n)
    # remove the 10 points with largest distance
    threshold = np.sort(distances)[-10]
    positions4 = positions2[distances < threshold, :] # removal
    #positions4 = positions2

    # again find x in Ax = B
    A = np.ones_like(positions4)
    A[:,:2] = positions4[:,:2]
    B = -positions4[:,2]
    x = np.linalg.inv(A.T @ A) @ A.T @ B
    n = x.copy()
    n[2] = 1
    positions2 = positions4

x_norm = n/np.linalg.norm(n)
ax.quiver(0, 0, 0, x_norm[0], x_norm[1], x_norm[2], length=1, normalize=True)
# print(x_norm)


# plot plane
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
fit = x
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = -(fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2])
ax.plot_wireframe(X,Y,Z, color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# apply rotation between x_norm and y-axis
z_axis = np.array([0, 0, 1])
rot_axis = np.cross(x_norm, z_axis)
rot_axis = rot_axis/np.linalg.norm(rot_axis)
angle = np.arccos(np.dot(x_norm, z_axis))
rot_matrix, _ = cv2.Rodrigues(angle*rot_axis)

positions3 = rot_matrix @ positions2.T
positions3 = positions3.T
ax.scatter(positions3[:,0], positions3[:,1], positions3[:,2], 'o')

# collect points again and app
positions = np.array([mat[:3,3] for mat in extrinsics])
positions4 = rot_matrix @ (positions - positions[np.invert(np.isnan(positions.sum(axis=1))),:].mean(axis=0)).T
positions4 = positions4.T
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions4[:,0], positions4[:,1], positions4[:,2], 'bo')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# # find center of rotation
# # TODO: remove outlier or using RANSAC
# from scipy import optimize
# method_2 = "leastsq"
# def calc_R(xc, yc):
#     """ calculate the distance of each 2D points from the center (xc, yc) """
#     return np.sqrt((x-xc)**2 + (y-yc)**2)

# def f_2(c):
#     """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
#     Ri = calc_R(*c)
#     return Ri - Ri.mean()

# x = positions4[np.invert(np.isnan(positions4.sum(axis=1))), 0]
# y = positions4[np.invert(np.isnan(positions4.sum(axis=1))), 1]
# center_estimate = x.mean(), y.mean()
# center_2, ier = optimize.leastsq(f_2, center_estimate)


# compute erros, mean and std
tilt_angles = []
tilt_angles2 = []
pan_angles = []
pan_angles2 = []
pan_step = 15.0
tilt_step = 15.0
pan_errors = []
tilt_errors = []
radii = []
text = []

############ Change this such that the first value of pan_angles[0] is close to zeros
#a = 118/180*np.pi #90/180*np.pi #-90/180*np.pi # 65/180*np.pi #
#a = 100/180*np.pi #110/180*np.pi #90/180*np.pi # -55/180*np.pi #
a=-180/180*np.pi
#####################
rotation_z = np.array([[np.cos(a), -np.sin(a), 0],
                       [np.sin(a), np.cos(a), 0],
                       [0, 0, 1]])
'''
for i in range(5):
    for j in range(24):
        pos = rotation_z @ positions4[i*24 + j, :]
        pos_norm = pos/np.linalg.norm(pos)
        tilt = np.arccos(np.dot(pos_norm, z_axis))
        pos_xy = pos[:2]
        pos_xy_norm = pos_xy/np.linalg.norm(pos_xy)
        pan = np.arccos(np.dot(pos_xy_norm, [0, 1]))
        tilt_angles.append(tilt/np.pi*180)
        tilt_angles2.append(60 + tilt_step*i)
        # tilt_errors.append(tilt/np.pi*180 - 120 + tilt_step*i)
        tilt_errors.append(tilt/np.pi*180 - 60 - tilt_step*i)
        pan_angles2.append(j*pan_step)
        if j <= 11: #10: # 12
            pan_angles.append(pan/np.pi*180)
            pan_errors.append(pan/np.pi*180 - j*pan_step)
        else:
            pan_angles.append(360-pan/np.pi*180)
            pan_errors.append(360-pan/np.pi*180 - j*pan_step)
        radii.append(np.linalg.norm(pos))
        text.append('%d, %d' % (i, j))
'''

for i in range(5):
    for j in range(24):
        pos = rotation_z @ positions4[i*24 + j, :]
        pos_norm = pos/np.linalg.norm(pos)
        tilt = np.arccos(np.dot(pos_norm, z_axis))
        pos_xy = pos[:2]
        pos_xy_norm = pos_xy/np.linalg.norm(pos_xy)
        pan = np.arccos(np.dot(pos_xy_norm, [0, 1]))
        tilt_angles.append(tilt/np.pi*180)
        tilt_angles2.append(60 + tilt_step*i)
        #tilt_errors.append(tilt/np.pi*180 - 120 + tilt_step*i)
        tilt_errors.append(tilt/np.pi*180 - 60 - tilt_step*i)
        pan_angles2.append(j*pan_step)
        if j <= 12:
            pan_angles.append(pan/np.pi*180)
            pan_errors.append(pan/np.pi*180 - j*pan_step)
        else:
            pan_angles.append(360-pan/np.pi*180)
            pan_errors.append(360-pan/np.pi*180 - j*pan_step)
        radii.append(np.linalg.norm(pos))
        text.append('%d, %d' % (i, j))
        
print('pan_angles[0] =', pan_angles[0])
print('tilt_angles[0] =', tilt_angles[0])

pan_errors = np.array(pan_errors)
pan_std = pan_errors[np.invert(np.isnan(pan_errors))].std()
print('pan_std [degree]:', pan_std)
pan_errors_mean = pan_errors[np.invert(np.isnan(pan_errors))].mean()
pan_errors = pan_errors - pan_errors_mean # remove a constant pan shift
pan_angles = pan_angles - pan_errors_mean
pan_angle_step = pan_angles[1:] - pan_angles[:-1]
pan_angle_step = pan_angle_step[pan_angle_step > 0] # remove wrong pairs
pan_angle_step_std = pan_angle_step.std()
pan_angle_step_mean = pan_angle_step.mean()
print('pan_angle_step_std [degree]:', pan_angle_step_std)
print('pan_angle_step_mean [degree]:', pan_angle_step_mean)

# exit()

tilt_errors = np.array(tilt_errors)
tilt_std = tilt_errors[np.invert(np.isnan(tilt_errors))].std()
print('tilt_std [degree]:', tilt_std)
tilt_errors_mean = tilt_errors[np.invert(np.isnan(tilt_errors))].mean()
tilt_errors = tilt_errors - tilt_errors_mean # remove a constant tilt shift
tilt_angles = tilt_angles - tilt_errors_mean
tilt_angle_step = tilt_angles[:-24] - tilt_angles[24:]
tilt_angle_step_std = tilt_angle_step[np.invert(np.isnan(tilt_angle_step))].std()
tilt_angle_step_mean = tilt_angle_step[np.invert(np.isnan(tilt_angle_step))].mean()
print('tilt_angle_step_std [degree]:', tilt_angle_step_std)
print('tilt_angle_step_mean [degree]:', tilt_angle_step_mean)

radii = np.array(radii)
radii_mean = radii[np.invert(np.isnan(radii))].mean()
radii_std = radii[np.invert(np.isnan(radii))].std()
radii_mean_mm = radii_mean * d0 / radii_mean
radii_std_mm = radii_std * d0 / radii_mean
print('radii_mean_mm:', radii_mean_mm)
print('radii_std_mm:', radii_std_mm)

# plot camera positions in physical dimension [mm]
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions4[:,0]*d0 / radii_mean, positions4[:,1]*d0 / radii_mean,
            positions4[:,2]*d0 / radii_mean, 'go')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([-220, 220])
ax.set_ylim([-220, 220])
ax.set_zlim([-180, 180])
plt.show()


plt.figure(5)
plt.plot(pan_angles, tilt_angles, 's')
for i in range(len(text)):
    plt.text(pan_angles[i], tilt_angles[i], text[i])
plt.plot(np.array([[i*15, i*15] for i in range(24)]).T,
          np.array([[60, 120] for i in range(24)]).T, '--k')
plt.xlabel('Pan angle  [degree]', fontsize=16)
plt.ylabel('Tilt angle [degree]', fontsize=16)

plt.figure(6)
plt.plot(pan_errors, tilt_errors, 's')
plt.xlabel('Pan angle error, STD=%0.3f [degree]' %pan_std, fontsize=16 )
plt.ylabel('Tilt angle error, STD=%0.3f [degree]' %tilt_std, fontsize=16 )
# plt.xlim(-15, 15)
# plt.ylim(-15, 15)

plt.show()

