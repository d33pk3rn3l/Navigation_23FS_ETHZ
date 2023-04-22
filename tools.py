#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:18:17 2023

@author: lukamueller
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt

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

    # !CHANGED: Second line is negative in order to change sign of y axis!
    C_bn = np.array([
        [cos_p*cos_y, -cos_r*sin_y+sin_r*sin_p*cos_y, sin_r*sin_y+cos_r*sin_p*cos_y],
        [-cos_p*sin_y, -cos_r*cos_y-sin_r*sin_p*sin_y, +sin_r*cos_y-cos_r*sin_p*sin_y],
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

def sensor_reduce(sensor,intv,sampling):
    """
    Shorten sensor time series
    
    ---
    Input:
        sensor   : python dictionary containing sensor data
        intv     : list with lower and upper limit for new time series in sec
        sampling : float, sampling step size
    ---
    Return:
        sensor : python dictionary containing reduced sensor data
    """
    
    for key in sensor:
        sensor[key] = sensor[key][int(intv[0]/sampling):int(intv[1]/sampling)]
    
    return(sensor)

def sensor_simulate(simulate,**kwargs):
    """
    Simulate accelerations and rotations 
    
    ---
    Input:
        simulate : python dictionary with parameters for INS data to be simulated
                    Example: 
                    simulate = {"accX":[[1,10,2]],
                                "accY":[],
                                "accZ":[],
                                "gyroZ":[[np.pi/2,20,4]]}
                    -> simulation of an acceleration in x direction of delta_v = 1 m/s, 
                       after 10 seconds with a duration of 2*2 seconds
                    -> simulation of a rotation about the z axis of delta_yaw = pi/2,
                       after 20 seconds with a duration of 2*4 seconds
    ---
    Return:
        sensor   : python dictionary containing reduced sensor data
    """

    sampling = kwargs.get('sampling',0.05)
    length = kwargs.get('length',180) # in sec
    time_start = kwargs.get('time_start',datetime.datetime(2000,1,1,0,0,0))
    
    time = np.arange(sampling,length,sampling)
    
    sensor = {}
    params = ["accX","accY","accZ","gyroX","gyroY","gyroZ","pitch","roll","yaw","lon","lat","altP"]
    
    #simulate accelerations and rotations for each parameter selected
    for para_cur in params:
        sensor[para_cur] = np.zeros(len(time))
        if not para_cur in simulate: continue
        for ele in simulate[para_cur]:
            sensor[para_cur] = sensor[para_cur] + ele[0]/(ele[2]*np.sqrt(2*np.pi)) * np.exp(-(time - ele[1])**2/(2*ele[2]**2))
        sensor[para_cur] = sensor[para_cur] + np.random.normal(0,0.0,len(sensor[para_cur]))
                
    # datetime array
    dt_lst = []
    for tim_cur in time:
        dt_lst.append(time_start + datetime.timedelta(seconds=float(tim_cur)))
    sensor["time"] = np.array(dt_lst) 
    
    return(sensor)

def sensor_plot(sensor,par_lst,name):
    """
    Plot selected parameters of sensor
    
    ---
    Input:
        sensor    : python dictionary containing sensor data
        par_lst   : list with names of parameters to be plotted, 
                    "Accelerometer", "Gyroscope" or "Location"
        name      : string, name extension of output figure file
    ---
    Return:
        -
    """
    
    if "Accelerometer" in par_lst:
        # Plot accelerometer data
        fig1, axs1 = plt.subplots(3, 1, sharex=True)
        fig1.suptitle('Accelerometer Data')
        axs1[0].plot(sensor["time"], sensor["accX"], color='#d7191c')
        axs1[0].set_ylabel('Acc. X [g]')
        axs1[1].plot(sensor["time"], sensor["accY"], color='#fee08b')
        axs1[1].set_ylabel('Acc. Y [g]')
        axs1[2].plot(sensor["time"], sensor["accZ"], color='#1a9850')
        axs1[2].set_ylabel('Acc. Z [g]')
        axs1[2].set_xlabel('Time')
        plt.tight_layout()
        plt.savefig("./output/accelerations_" + name + ".png",dpi=200)
        
    if "Gyroscope" in par_lst:
        # Plot gyroscope data
        fig1, axs1 = plt.subplots(3, 1, sharex=True)
        fig1.suptitle('Gyroscope Data')
        axs1[0].plot(sensor["time"], sensor["gyroX"], color='#d7191c')
        axs1[0].set_ylabel('Rot. X [rad/s]')
        axs1[1].plot(sensor["time"], sensor["gyroY"], color='#fee08b')
        axs1[1].set_ylabel('Rot. Y [rad/s]')
        axs1[2].plot(sensor["time"], sensor["gyroZ"], color='#1a9850')
        axs1[2].set_ylabel('Rot. Z [rad/s]')
        axs1[2].set_xlabel('Time')
        plt.tight_layout()
        plt.savefig("./output/rotations_" + name + ".png",dpi=200)

    if "Location" in par_lst:
        # Plot location data
        fig1, axs1 = plt.subplots(3, 1, sharex=True)
        fig1.suptitle('Locations')
        axs1[0].plot(sensor["time"], sensor["lon"], color='#d7191c')
        axs1[0].set_ylabel('Longitude')
        axs1[1].plot(sensor["time"], sensor["lat"], color='#fee08b')
        axs1[1].set_ylabel('Latitude')
        axs1[2].plot(sensor["time"], sensor["altP"], color='#1a9850')
        axs1[2].set_ylabel('Altitude')
        axs1[2].set_xlabel('Time')
        plt.tight_layout()
        plt.savefig("./output/location_" + name + ".png",dpi=200)
