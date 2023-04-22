# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate
import datetime
import csv
import os

"""
1. read_csv_sensor_logger -----------------------------------------------------
"""

def read_csv_sensor_logger(directory,folder,**kwargs):

    """
    Read csv files from the app Sensor Logger.
    
    Data are transformed from the "device system" into the body system with:
        x_b = y_d
        y_b = -x_d
        z_b = z_d
        
    Signs of accX, accY and accZ are opposite to the accalerations measured 
    by the device
   
    ---
    Input:
        directory   : string, directory where data is located
        folder      : string, name of folder exported from Sensor Logger app
        g           : float, gravity at the measurement site
        sampling    : float, sampling step size
        skip_lines  : int, number of epochs to skip at the beginning
    ---
    Return:
        sensor : python dictionary, keys: time, lon, lat, atlP, accX, accY, accZ,
                                          gyroX, gyroY, gyroZ, pitch, roll, yaw
    """
    
    # default parameter values
    g = kwargs.get('g',9.81)
    sampling = kwargs.get('sampling',0.05)
    skip_lines = kwargs.get('skip_lines',0)
    
    # initialize sensor and dir_csv
    sensor = {}
    dir_csv = os.path.join(directory,folder)
    
    if os.path.exists(os.path.join(dir_csv,"TotalAcceleration.csv")):
        # read total accelerations
        data = np.loadtxt(os.path.join(dir_csv,"TotalAcceleration.csv"),delimiter=",",skiprows=1+skip_lines)
        tim_new = np.arange(data[0,1],data[-1,1],sampling)
        header_acc= next(csv.reader(open(os.path.join(dir_csv,"TotalAcceleration.csv"),newline='')))
        data_acc = np.zeros((len(tim_new),np.shape(data)[1]))
        
        for i_col in range(np.shape(data)[1]):
            f_int = interpolate.interp1d(data[:,1], data[:,i_col],kind="linear",fill_value="extrapolate")
            data_acc[:,i_col] = f_int(tim_new)
        
        # compute F_g
        data_Fg = -1 * data_acc / g # flip sign
    else:
        # read accelerations due to gravity
        data = np.loadtxt(os.path.join(dir_csv,"Gravity.csv"),delimiter=",",skiprows=1 + skip_lines)
        tim_new = np.arange(data[0,1],data[-1,1],sampling)
        data_grav = np.zeros((len(tim_new),np.shape(data)[1]))
        
        for i_col in range(np.shape(data)[1]):
            f_int = interpolate.interp1d(data[:,1], data[:,i_col],kind="linear",fill_value="extrapolate")
            data_grav[:,i_col] = f_int(tim_new)
         
        # read linear accelerations
        data = np.loadtxt(os.path.join(dir_csv,"Accelerometer.csv"),delimiter=",",skiprows=1)
        header_acc= next(csv.reader(open(os.path.join(dir_csv,"Accelerometer.csv"),newline='')))
        data_acc = np.zeros((len(tim_new),np.shape(data)[1]))
        
        for i_col in range(np.shape(data)[1]):
            f_int = interpolate.interp1d(data[:,1], data[:,i_col],kind="linear",fill_value="extrapolate")
            data_acc[:,i_col] = f_int(tim_new)
        
        # compute F_g
        data_Fg = data_grav # flip sign
        data_Fg[:,2:] = -1 * (data_Fg[:,2:] + data_acc[:,2:]) / g

    sensor["accX"] = data_Fg[:,header_acc.index('y')]
    sensor["accY"] = -1 * data_Fg[:,header_acc.index('x')]
    sensor["accZ"] = data_Fg[:,header_acc.index('z')]
    
    # read gyroscope data
    data = np.loadtxt(os.path.join(dir_csv,"Gyroscope.csv"),delimiter=",",skiprows=1)
    header_gyro= next(csv.reader(open(os.path.join(dir_csv,"Gyroscope.csv"),newline='')))
    data_gyro = np.zeros((len(tim_new),np.shape(data)[1]))
    
    for i_col in range(np.shape(data)[1]):
        f_int = interpolate.interp1d(data[:,1], data[:,i_col],kind="linear",fill_value="extrapolate")
        data_gyro[:,i_col] = f_int(tim_new)
    
    sensor["gyroX"] = data_gyro[:,header_gyro.index('y')]
    sensor["gyroY"] = -1 * data_gyro[:,header_gyro.index('x')]
    sensor["gyroZ"] = data_gyro[:,header_gyro.index('z')]
        
    # read location data
    fil_loc = os.path.join(dir_csv,"Location.csv")
    if os.path.exists(os.path.join(dir_csv,"LocationGps.csv")) and os.path.getsize(os.path.join(dir_csv,"LocationGps.csv")) > 0:
        fil_loc = os.path.join(dir_csv,"LocationGps.csv")
        
    data = np.loadtxt(fil_loc,delimiter=",",skiprows=1)
    header_loc= next(csv.reader(open(os.path.join(dir_csv,"Location.csv"),newline='')))
    data_loc = np.zeros((len(tim_new),np.shape(data)[1]))
    
    for i_col in range(np.shape(data)[1]):
        f_int = interpolate.interp1d(data[:,1], data[:,i_col],kind="linear",fill_value="extrapolate")
        data_loc[:,i_col] = f_int(tim_new)    
    
    sensor["lat"] = data_loc[:,header_loc.index('latitude')]
    sensor["lon"] = data_loc[:,header_loc.index('longitude')]
    sensor["altP"] = data_loc[:,header_loc.index('altitude')]
    
    # read time
    date_start = datetime.datetime.strptime(folder, '%Y-%m-%d_%H-%M-%S')
    dt_lst = []
    
    for i_tim,tim_cur in enumerate(tim_new):
       dt_lst.append(date_start + datetime.timedelta(seconds=tim_new[i_tim]))
    
    sensor["time"] = np.array(dt_lst)   
    
    # read Orientation
    data = np.loadtxt(os.path.join(dir_csv,"Orientation.csv"),delimiter=",",skiprows=1)
    header_ori = next(csv.reader(open(os.path.join(dir_csv,"Orientation.csv"),newline='')))
    data_ori = np.zeros((len(tim_new),np.shape(data)[1]))
    
    for i_col in range(np.shape(data)[1]):
        f_int = interpolate.interp1d(data[:,1], data[:,i_col],kind="linear",fill_value="extrapolate")
        data_ori[:,i_col] = f_int(tim_new)    
    
    sensor["pitch"] = data_ori[:,header_ori.index('pitch')]
    sensor["roll"] = data_ori[:,header_ori.index('roll')]
    sensor["yaw"] = -1 * data_ori[:,header_ori.index('yaw')]
    

    return(sensor)

"""
2. read_csv_physics_toolbox ---------------------------------------------------
"""

def read_csv_physics_toolbox(directory, file_bn,**kwargs):

    """
    Read csv file from the app Physics Toolbox Sensor Suite.
    
    Data are transformed from the "device system" into the body system with:
        x_b = y_d
        y_b = -x_d
        z_b = z_d
        
    Signs of accX, accY and accZ are opposite to the accalerations measured 
    by the device
   
    ---
    Input:
        directory   : string, directory where data is located
        folder      : string, name of folder exported from Sensor Logger app
        sampling    : float, sampling step size
        P_0         : float, air pressure at sea level (hPa)
        temp        : float, temperature at measurement site
        skip_lines  : int, number of epochs to skip at the beginning
    ---
    Return:
        sensor : python dictionary, keys: time, lon, lat, atlP, accX, accY, accZ,
                                          gyroX, gyroY, gyroZ, pitch, roll, yaw
    """
    
    # default parameter values
    sampling = kwargs.get('sampling',0.05)
    P_0 = kwargs.get('P_0',1013.25)
    temp = kwargs.get('temp',10)
    skip_lines = kwargs.get('skip_lines',0)
    
    # initialize sensor and dir_csv
    sensor = {}
    file = os.path.join(directory,file_bn)
    
    # replace comma by dot
    with open(file,"r") as fil_hdl:
    
        fil_str = fil_hdl.read()
        fil_str = fil_str.strip().replace(',','.')
    
    # write new file
    with open("temp.csv","w") as fil_tmp:
        fil_tmp.write("\n" + fil_str)   
    os.rename("temp.csv",file)
    
    # get header
    with open(file,"r") as csv_hdl: csv_lins = csv_hdl.readlines()
    header = csv_lins[1].split(";")
    
    data = np.loadtxt(file,delimiter=";",skiprows=2+skip_lines,usecols = range(len(header)-1))
    tim_new = np.arange(data[0,0],data[-1,0],sampling)
    data_sens = np.zeros((len(tim_new),np.shape(data)[1]))
    
    for i_col in range(np.shape(data)[1]):
        f_int = interpolate.interp1d(data[:,0], data[:,i_col],kind="linear",fill_value="extrapolate")
        data_sens[:,i_col] = f_int(tim_new)
    
    # fill variable "sensor", sign flipped!
    sensor["accX"] = -1 * data_sens[:,header.index('gFy')]
    sensor["accY"] = data_sens[:,header.index('gFx')]
    sensor["accZ"] = -1 * data_sens[:,header.index('gFz')]
 
    sensor["gyroX"] = data_sens[:,header.index('wy')]
    sensor["gyroY"] = -1 * data_sens[:,header.index('wx')]
    sensor["gyroZ"] = data_sens[:,header.index('wz')]

    sensor["pitch"] = data_sens[:,header.index('Pitch')]*np.pi/180 # values given in degrees
    sensor["roll"] = -1 * data_sens[:,header.index('Roll')]*np.pi/180
    sensor["yaw"] = -1 * data_sens[:,header.index('Azimuth')]*np.pi/180
    
    sensor["lon"] = data_sens[:,header.index('Longitude')]
    sensor["lat"] = data_sens[:,header.index('Latitude')]
    P = data_sens[:,header.index('p')]
    P[P==0] = np.nan
    # use air pressure to compute height component, since height is not recorded
    sensor["altP"] = ((P_0/P)**(1/5.275) - 1) * (temp + 273.15) / 0.0065
    
    # read time
    date_start = datetime.datetime.strptime(file_bn, '%Y-%m-%d%H.%M.%S.csv')
    dt_lst = []
    
    for i_tim,tim_cur in enumerate(tim_new):
       dt_lst.append(date_start + datetime.timedelta(seconds=tim_new[i_tim]))
    sensor["time"] = np.array(dt_lst) 
    
    return(sensor)

read_csv_physics_toolbox

"""
3. read_csv_sensorlog ---------------------------------------------------------
"""

def read_csv_sensorlog(directory, file_bn,**kwargs):

    """
    Read csv file from the app Sensorlog.

    Data are transformed from the "device system" into the body system with:
    x_b = y_d
    y_b = -x_d
    z_b = z_d
        
    ---
    Input:
        sampling    : float, sampling step size
        skip_lines  : int, number of epochs to skip at the beginning
    ---
    Return:
        sensor : python dictionary, keys: time, lon, lat, atlP, accX, accY, accZ,
                                          gyroX, gyroY, gyroZ, pitch, roll, yaw
    """
    
    # default sampling
    sampling = kwargs.get('sampling',0.05)
    skip_lines = kwargs.get('skip_lines',0)
    
    # initialize sensor and dir_csv
    sensor = {}
    file = os.path.join(directory,file_bn)
    
    # get header
    with open(file,"r") as csv_hdl: csv_lins = csv_hdl.readlines()
    header = csv_lins[0].split(",")
    
    data = np.genfromtxt(file,delimiter=",",skip_header=1+skip_lines,usecols = range(len(header)))
    tim_new = np.arange(data[0,header.index("accelerometerTimestamp_sinceReboot(s)")].astype(float),data[-1,header.index("accelerometerTimestamp_sinceReboot(s)")].astype(float),sampling)
    
    data_sens = np.zeros((len(tim_new),np.shape(data)[1]))
    
    # interpolation 
    for i_col in range(1,np.shape(data)[1]):
        f_int = interpolate.interp1d(data[:,header.index("accelerometerTimestamp_sinceReboot(s)")].astype(float), data[:,i_col].astype(float),kind="linear",fill_value="extrapolate")
        data_sens[:,i_col] = f_int(tim_new)
    
    # fill variable "sensor"
    sensor["accX"] = data_sens[:,header.index('accelerometerAccelerationY(G)')]
    sensor["accY"] = -1*data_sens[:,header.index('accelerometerAccelerationX(G)')]
    sensor["accZ"] = data_sens[:,header.index('accelerometerAccelerationZ(G)')]

    sensor["gyroX"] = data_sens[:,header.index('gyroRotationY(rad/s)')]
    sensor["gyroY"] = -1*data_sens[:,header.index('gyroRotationX(rad/s)')]
    sensor["gyroZ"] = data_sens[:,header.index('gyroRotationZ(rad/s)')]
    
    sensor["pitch"] = -1*data_sens[:,header.index('motionPitch(rad)')]
    sensor["roll"] = data_sens[:,header.index('motionRoll(rad)')]
    sensor["yaw"] = data_sens[:,header.index('motionYaw(rad)')]

    sensor["lon"] = data_sens[:,header.index('locationLongitude(WGS84)')]
    sensor["lat"] = data_sens[:,header.index('locationLatitude(WGS84)')]
    sensor["altP"] = data_sens[:,header.index('locationAltitude(m)')]
    
    # read time
    date_start = datetime.datetime.strptime(file_bn,'%Y-%m-%d_%H_%M_%S_my_iOS_device.csv')
    dt_lst = []
    
    for i_tim,tim_cur in enumerate(tim_new):
       dt_lst.append(date_start + datetime.timedelta(seconds=tim_new[i_tim]))
    sensor["time"] = np.array(dt_lst) 

    return(sensor)
    
    
