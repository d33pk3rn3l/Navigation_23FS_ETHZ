#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:09:39 2023

@author: lukamueller
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import cal_initial_roll_pitch_yaw, write_kml_file, sensor_reduce, sensor_plot, sensor_simulate, rot_b_n
from read_sensor_output import read_csv_physics_toolbox,read_csv_sensor_logger, read_csv_sensorlog

# add directory of main.py to path
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


sampling = 0.05 # sampling step size (s)
init_intv = [2500, 5000] # interval to compute initial orientation
skip_lines = 30 # number of epochs to skip at the beginning of the recording

# data sensor_logger
sensor = read_csv_sensor_logger(os.path.join(file_dir,"data"), "2023-03-23_07-50-30",
                                sampling=sampling,
                                skip_lines=skip_lines) # plt.2023-03-19_09-36-59

print(sensor["accX"][init_intv[0]:init_intv[1]].std())
print(sensor["accY"][init_intv[0]:init_intv[1]].std())
print(sensor["accZ"][init_intv[0]:init_intv[1]].std())

#pd.DataFrame(sensor).to_csv(os.path.join(file_dir,"output/sensor_logger_raw_data.csv"))

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
plt.savefig("./output/accelerations.png",dpi=200)

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
plt.savefig("./output/rotations.png",dpi=200)

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
plt.savefig("./output/location.png",dpi=200)

write_kml_file(sensor["lon"][10:],sensor["lat"][10:],sensor["altP"][10:],"./output/gps_location.kml")

r0,p0,y0 = cal_initial_roll_pitch_yaw(sensor,init_intv[0],init_intv[1]) 

# Print results from initial orientation
print("Pitch: {0:3.4f} (computed), {1:3.4f} (recorded)".format(p0,np.mean(sensor["pitch"][init_intv[0]:init_intv[1]])))
print("Roll: {0:3.4f} (computed), {1:3.4f} (recorded)".format(r0,np.mean(sensor["roll"][init_intv[0]:init_intv[1]])))
print("Yaw: {0:3.4f} (computed), {1:3.4f} (recorded)".format(y0,np.mean(sensor["yaw"][init_intv[0]:init_intv[1]])))

# Reduce data to first interessting part
START = 580 # seconds
END = 800 # seconds (1100 is the end of the interesting part)
sensor_reduced = sensor_reduce(sensor, [START, END], sampling)

# Plot reduced data
sensor_plot(sensor_reduced, ["Gyroscope"], "reduced")

# Preparation
GRAVITY = 9.81
EARTH_RADIUS = 6371000 # m
sensor_reduced["accX"] *= -GRAVITY
sensor_reduced["accY"] *= -GRAVITY
sensor_reduced["accZ"] *= -GRAVITY
gravity_vector = np.array([0, 0, GRAVITY])

# Initialization
init_position = np.array([np.mean(sensor_reduced["lat"][:100]),
                 np.mean(sensor_reduced["lon"][:100]),
                 np.mean(sensor_reduced["altP"][:100])])
init_velocity = np.array([0, 0, 0])
init_orientation = np.array([r0, p0, y0])

# Integration of rotations and accelerations
def integrate_rotations_and_accelerations(sensor_data, init_position, init_velocity, init_orientation, sampling):
    num_data_points = len(sensor_data["time"])
    position = np.zeros((num_data_points, 3)) # [lat, lon, alt]
    velocity = np.zeros((num_data_points, 3)) # [v_n, v_e, v_d]
    orientation = np.zeros((num_data_points, 3)) # [roll, pitch, yaw] in navigation frame

    # Set initial values
    position[0] = init_position
    velocity[0] = init_velocity
    orientation[0] = init_orientation

    for i in range(1, num_data_points):
        step_dt = sampling

        # Calculate orientation for transformation from body frame to navigation frame
        rotation_rates = np.array([ sensor_data["gyroX"][i],
                                    sensor_data["gyroY"][i],
                                    sensor_data["gyroZ"][i]])
        orientation[i] = orientation[i - 1] + rotation_rates * step_dt

        # Calculate transformation matrix from body frame to navigation frame for current step
        C_bn = rot_b_n(orientation[i][0], orientation[i][1], orientation[i][2])
        
        # Transform acceleration from body frame to navigation frame
        acceleration_nav = np.dot(C_bn,
                                  np.array([sensor_data["accX"][i],
                                            sensor_data["accY"][i],
                                            sensor_data["accZ"][i]]))
        
        # Subtract gravity vector
        acceleration_nav_corrected = acceleration_nav - gravity_vector

        # Calculate velocity
        velocity[i] = velocity[i - 1] + acceleration_nav_corrected * step_dt

        # Update position with quick and dirty transformation to WGS84
        dlat = velocity[i][0] * step_dt / EARTH_RADIUS # quick and dirty approximation, since movement is small
        dlon = velocity[i][1] * step_dt / (EARTH_RADIUS * np.cos(np.deg2rad(position[i - 1][0]))) # d_e / R * cos(lat)
        dalt = -velocity[i][2] * step_dt  # Subtracting because positive velocity[i][2] is in the downward direction
        position[i] = position[i - 1] + np.array([np.rad2deg(dlat), np.rad2deg(dlon), dalt])

    return position, velocity, orientation

position, velocity, orientation = integrate_rotations_and_accelerations(sensor_reduced,
                                                                        init_position,
                                                                        init_velocity,
                                                                        init_orientation,
                                                                        sampling)

# Plot position
# Assuming `position` is a NumPy array with shape (num_data_points, 3) containing the position data
lat = position[:, 0]
lon = position[:, 1]
alt = position[:, 2]

# Create the scatter plot
title = "Position Scatter Plot (Period: " + str(END-START) + " seconds)"
plt.figure()
plt.scatter(lon, lat, c=alt, cmap='viridis', marker='.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(title)
plt.colorbar(label='Altitude')
plt.savefig("./output/position_scatter_plot.png", dpi=200)
plt.xlim([np.min(lon), np.max(lon)])
plt.ylim([np.min(lat), np.max(lat)])
plt.show()
