#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:09:39 2023

@author: lukamueller
"""

import matplotlib.pyplot as plt 
import numpy as np
import os
import sys

# add directory of main.py to path 
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from tools import cal_initial_roll_pitch_yaw,write_kml_file
from read_sensor_output import read_csv_physics_toolbox,read_csv_sensor_logger,read_csv_sensorlog

sampling = 0.05 # sampling step size (s)
init_intv = [1000,3000] # interval to compute initial orientation
skip_lines = 30 # number of epochs to skip at the beginning of the recording

# data sensor_logger
sensor = read_csv_sensor_logger(os.path.join(file_dir,"data"), "2023-03-23_07-50-30", sampling=sampling, skip_lines=skip_lines) # plt.2023-03-19_09-36-59

# data physics toolbox
#sensor = read_csv_physics_toolbox(os.path.join(file_dir,"data"),"2023-03-1717.06.59.csv", sampling=sampling,skip_lines=skip_lines)

# data sensorlog
#sensor = read_csv_sensorlog(os.path.join(file_dir,"data"),"2021-04-13_18_18_36_my_iOS_device.csv", sampling=sampling,skip_lines=skip_lines)

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
plt.savefig("./Python/output/accelerations.png",dpi=200)

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
plt.savefig("./Python/output/rotations.png",dpi=200)

# Plot location data
fig1, axs1 = plt.subplots(3, 1, sharex=True)
fig1.suptitle('Location Data')
axs1[0].plot(sensor["time"], sensor["lon"], color='#d7191c')
axs1[0].set_ylabel('Longitude')
axs1[1].plot(sensor["time"], sensor["lat"], color='#fee08b')
axs1[1].set_ylabel('Latitude')
axs1[2].plot(sensor["time"], sensor["altP"], color='#1a9850')
axs1[2].set_ylabel('Altitude')
axs1[2].set_xlabel('Time')
plt.tight_layout()
plt.savefig("./Python/output/location.png",dpi=200)

write_kml_file(sensor["lon"][10:],sensor["lat"][10:],sensor["altP"][10:],"./output/gps_location.kml")

r0,p0,y0 = cal_initial_roll_pitch_yaw(sensor,init_intv[0],init_intv[1]) 

# Print results from initial orientation
print("Pitch: {0:3.4f} (computed), {1:3.4f} (recorded)".format(p0,np.mean(sensor["pitch"][init_intv[0]:init_intv[1]])))
print("Roll: {0:3.4f} (computed), {1:3.4f} (recorded)".format(r0,np.mean(sensor["roll"][init_intv[0]:init_intv[1]])))
print("Yaw: {0:3.4f} (computed), {1:3.4f} (recorded)".format(y0,np.mean(sensor["yaw"][init_intv[0]:init_intv[1]])))