#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:18:17 2023

@author: lukamueller
"""

import numpy as np

def write_kml_file(lon,lat,alt,file_out):
    
    coor = np.stack((lon,lat,alt)).transpose()
    str_coor = ''
    for coor_cur in coor:
        str_coor = str_coor + ",".join(str(i) for i in coor_cur.tolist()) + "\n"
        
    with open("template.kml","r") as kml_hdl:
        kml_str = kml_hdl.read()
        
    kml_str_new = kml_str.replace("coordinates_sensor",str_coor)
    
    with open(file_out,"w") as kml_new:
        kml_new.write(kml_str_new)

def rot_b_n(r, p, y):
    """
    The function rot_b_n computes the rotation matrix for the transformation
    from the body frame into the navigation frame.

    Inputs:
    r, p, y: roll, pitch, and yaw angles in radians

    Outputs:
    C_bn: 3 x 3 transformation matrix
    """
    cos_r = np.cos(r)
    sin_r = np.sin(r)
    cos_p = np.cos(p)
    sin_p = np.sin(p)
    cos_y = np.cos(y)
    sin_y = np.sin(y)

    C_bn = np.array([
        [cos_p*cos_y, -cos_r*sin_y+sin_r*sin_p*cos_y, sin_r*sin_y+cos_r*sin_p*cos_y],
        [cos_p*sin_y, cos_r*cos_y+sin_r*sin_p*sin_y, -sin_r*cos_y+cos_r*sin_p*sin_y],
        [-sin_p,      sin_r*cos_p,                     cos_r*cos_p]
    ])

    return C_bn
        
def cal_initial_roll_pitch_yaw(sensor,navg1,navg2):
    
    # Extract and average data
    # accelerometer data [m/s^2]
    accX = np.mean(sensor['accX'][navg1:navg2])
    accY = np.mean(sensor['accY'][navg1:navg2])
    accZ = np.mean(sensor['accZ'][navg1:navg2])
    # gyroscope data [rad/s]
    gyroX = np.mean(sensor['gyroX'][navg1:navg2])
    gyroY = np.mean(sensor['gyroY'][navg1:navg2])
    gyroZ = np.mean(sensor['gyroZ'][navg1:navg2])

    # Compute roll r0 and pitch p0 [rad]
    r0 = np.arctan(accY / accZ)
    p0 = np.arctan(accX / np.sqrt(accY**2 + accZ**2))

    # Transform gyroscope measurements from body frame into navigation frame
    # transformation matrix C_bn
    C_bn = rot_b_n(r0, p0, 0)
    gyro_n = np.dot(C_bn, np.array([gyroX, gyroY, gyroZ]))

    # Compute yaw y0 [rad]
    y0 = np.arctan2(-gyro_n[1], gyro_n[0])

    return r0, p0, y0