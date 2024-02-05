import numpy as np
import pickle
import math

def lla_to_enu(C_lat, C_lon, C_alt, O_lat, O_lon, O_alt):
    '''
        Calculate the relative location with respect to the selected origin in the local ENU coordinate. unit: meter
        C_lat, C_lon, C_alt: current location
        O_lat, O_lon, O_alt: origin location
    '''

    Ea = 6378137
    Eb = 6356752.3142
    C_lat = math.radians(C_lat)
    C_lon = math.radians(C_lon)
    O_lat = math.radians(O_lat)
    O_lon = math.radians(O_lon)
    Ec = Ea * (1 - (Ea - Eb) / Ea * (math.sin(C_lat)) ** 2) + C_alt
    d_lat = C_lat - O_lat
    d_lon = C_lon - O_lon
    e = d_lon * Ec * math.cos(C_lat)
    n = d_lat * Ec
    u = C_alt - O_alt
    return e, n, u



if __name__ == "__main__":
    file_path = 'I:/dataset-final/train/2023-04-08-02-33-11-1-conti/loc_pose_vel.pkl'
    with open(file_path, 'rb') as f:
        loc_pose_vel = pickle.load(f)

    # we take the first location as origin
    lat_origin = loc_pose_vel[0]["lat"]
    lon_origin = loc_pose_vel[0]["lon"]
    alt_origin = loc_pose_vel[0]["alt"]/1000  # mm --> m
    e, n, u = lla_to_enu(loc_pose_vel[4]["lat"], loc_pose_vel[4]["lon"], loc_pose_vel[4]["alt"]/1000, lat_origin, lon_origin, alt_origin)
