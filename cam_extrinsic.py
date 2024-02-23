import numpy as np
import math
import pickle
import os
import open3d as o3d
from projection import read_calib_params
from read_imu_rtk import lla_to_enu

class Extrinsic(object):
    def __init__(self, is_half=True):
        self.calib_path = 'I:/data/dev_kit/calibration/'

        self.R_enu2ori = np.empty(shape=(3, 3))
        self.T_enu2ori = np.empty(shape=(3, 1))
        self.R_equ = np.empty(shape=( 3, 3))
        self.T_equ = np.empty(shape=(3, 1))

        # pre_read the extrinsic parameters between camera and lidar
        # intrinsics (after rectification): calib_params["K"]
        # stereo baseline(in mm): calib_params["B"]
        # lidar -> left camera extrinsics: calib_params["R"], calib_params["T"]
        if is_half:
            calib_files = ['calib_20230317_half.pkl', 'calib_20230321_half.pkl', 'calib_20230406_half.pkl', 'calib_20230408_half.pkl', 'calib_20230409_half.pkl']
        else:
            calib_files = ['calib_20230317.pkl', 'calib_20230321.pkl', 'calib_20230406.pkl',
                           'calib_20230408.pkl', 'calib_20230409.pkl']
        self.calib_params_all = {}
        for file in calib_files:
            calib_params = read_calib_params(os.path.join(self.calib_path, file))
            calib_params['R_inv'] = np.linalg.inv(calib_params['R'])
            date = file[6:14]
            self.calib_params_all[date] = calib_params

    def get_calib(self, date_stamp):
        # name in format: 20230408023213.400
        date = date_stamp[:8]
        return self.calib_params_all[date]

    def yaw_convert(self, yaw):
        '''
            convert the yaw data from [0, 360] to [-180, 180]
        '''
        if np.pi <= yaw <= 2 * np.pi:
            yaw -= 2 * np.pi
        return yaw

    def get_RT_lidar(self, sample_cur: dict, sample_ref: dict):
        '''
            As the location and orientation is given in LiDRA's coordinate, the camera extrinsic is represented and transformed from lidar's extrinsic
        '''
        #### pre-process
        rotX_cur = 0.017453 * sample_cur['pitch']
        rotY_cur = 0.017453 * sample_cur['roll']
        rotZ_cur = 0.017453 * sample_cur['yaw']
        rotZ_cur = self.yaw_convert(rotZ_cur)
        lat_cur = sample_cur["lat"]
        lon_cur = sample_cur["lon"]
        alt_cur = sample_cur["alt"] / 1000  # mm --> m

        # the transformation from current lidar to lidar ENU, rotation order in ZXY
        R_X1 = np.array([[1, 0, 0], [0, np.cos(rotX_cur), -np.sin(rotX_cur)], [0, np.sin(rotX_cur), np.cos(rotX_cur)]])
        R_Y1 = np.array([[np.cos(rotY_cur), 0, np.sin(rotY_cur)], [0, 1, 0], [-np.sin(rotY_cur), 0, np.cos(rotY_cur)]])
        R_Z1 = np.array([[np.cos(rotZ_cur), -np.sin(rotZ_cur), 0], [np.sin(rotZ_cur), np.cos(rotZ_cur), 0], [0, 0, 1]])
        self.R_cur2enu = np.matmul(R_Z1, np.matmul(R_X1, R_Y1))   # the rotation from current lidar to enu

        rotX_ori = 0.017453 * sample_ref['pitch']
        rotY_ori = 0.017453 * sample_ref['roll']
        rotZ_ori = 0.017453 * sample_ref['yaw']
        rotZ_ori = self.yaw_convert(rotZ_ori)
        lat_ori = sample_ref["lat"]
        lon_ori = sample_ref["lon"]
        alt_ori = sample_ref["alt"] / 1000

        # the translation from current lidar ENU to origin's ENU
        self.T_enu2ori[:, 0] = lla_to_enu(lat_cur, lon_cur, alt_cur, lat_ori, lon_ori, alt_ori)

        # the rotation from origin ENU to real orientation
        R_X2 = np.array([[1, 0, 0], [0, np.cos(rotX_ori), np.sin(rotX_ori)], [0, -np.sin(rotX_ori), np.cos(rotX_ori)]])
        R_Y2 = np.array([[np.cos(rotY_ori), 0, -np.sin(rotY_ori)], [0, 1, 0], [np.sin(rotY_ori), 0, np.cos(rotY_ori)]])
        R_Z2 = np.array([[np.cos(rotZ_ori), np.sin(rotZ_ori), 0], [-np.sin(rotZ_ori), np.cos(rotZ_ori), 0], [0, 0, 1]])
        self.R_enu2ori = np.matmul(R_Y2, np.matmul(R_X2, R_Z2))

    def get_cam_extr(self, sample_cur, sample_ref):
        l2c_calib = self.get_calib(sample_cur['time'])
        self.get_RT_lidar(sample_cur, sample_ref)
        # the transformation above is conducted under the LiDAR's coordinate.

        T_l2c = l2c_calib['T']
        R_l2c = l2c_calib['R']
        # calculate the equivalent rotation matrix
        R_temp = R_l2c @ self.R_enu2ori @ self.R_cur2enu @ l2c_calib['R_inv']
        R_equ = R_temp
        # calculate the equivalent translation matrix
        T_equ = T_l2c - R_temp @ T_l2c + R_l2c @ self.R_enu2ori @ self.T_enu2ori
        
        return R_equ, T_equ


if __name__ == '__main__':
    file_path = 'I:/data/RSRD-dense/train/2023-04-09-02-07-24-10-conti/loc_pose_vel.pkl'
    with open(file_path, 'rb') as f:
        loc_pose_vel = pickle.load(f)
    
    # here we take the first as origin, calculate the camera extrinsic of current-->origin 
    current = loc_pose_vel[4]
    reference = loc_pose_vel[0]
    extrinsic = Extrinsic(is_half=True)
    R_equ, T_equ = extrinsic.get_cam_extr(current, reference)
    # the point [x,y,z] in the current camera's coordinate can be represented under the origin coordinate
    # researchers can also get the other forms such as Euler angle and Quaternion
    p_cur = [1, 1, 1]
    p_ori = np.matmul(R_equ, p_cur) + T_equ
