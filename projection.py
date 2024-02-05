import numpy as np
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

def read_calib_params(calib_file):
    '''
         intrinsics (after rectification): calib_params["K"]
         stereo baseline(in mm): calib_params["B"]
         lidar -> left camera extrinsics: calib_params["R"], calib_params["T"]
     '''
    with open(calib_file, 'rb') as f:
        calib_params = pickle.load(f)

    return calib_params

def project_point2camera(calib_params, cloud):
    # then the point cloud can be projected onto the image plane by the cv2 function: cv2.projectPoints()
    # here, we calculate manually:
    point_uv = np.zeros((cloud.shape[0], 2))
    point_camera_depth = np.zeros((cloud.shape[0],))
    for i in range(cloud.shape[0]):
        point_lidar = np.transpose(cloud[i:i+1, :])  # (3, 1)
        point_camera = np.matmul(calib_params["R"], point_lidar) + calib_params["T"]  # from lidar coord --> rectified camera coord
        point_camera_depth[i] = point_camera[2]
        uv_pixel = np.matmul(calib_params["K"], point_camera)  # from rectified camera coord --> pixel coord
        point_uv[i, 0] = uv_pixel[0] / uv_pixel[2]
        point_uv[i, 1] = uv_pixel[1] / uv_pixel[2]

    # then, preserve the points in the camera's perspective
    u, v, depth_uv = [], [], []  # (u,v) is the (width, height) coordinate of the projected point，depth is the corresponding depth
    for point, depth in zip(point_uv, point_camera_depth):
        u_temp = point[0]
        v_temp = point[1]
        if 0 <= u_temp <= calib_params['Width'] and 0 <= v_temp <= calib_params['Height']:
            u.append(u_temp)
            v.append(v_temp)
            depth_uv.append(depth)

    u = np.array(u)
    v = np.array(v)
    u = np.expand_dims(u, axis=1)
    v = np.expand_dims(v, axis=1)
    depth_uv = np.array(depth_uv, dtype=np.float32)

    return np.concatenate((u, v), axis=1), depth_uv


def show_image_with_points(uv, depth_uv, image):
    # show the image with projected lidar points
    plt.figure(dpi=300, figsize=(10, 6))
    plt.scatter(uv[:, 0], uv[:, 1], c=depth_uv, cmap='brg', s=0.1, alpha=0.5)
    plt.colorbar()
    plt.imshow(image)
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def get_dis_depth_map(uv, depth_uv, calib_params):
    # the uv location is coarsely converted to integer here
    # when building the dataset, we 'round' the coordinate value and average the depth values in the same pixel.
    uv_int = uv.astype(np.int16)

    dis_map = np.zeros((calib_params['Height'], calib_params['Width']), dtype=np.float32)  # disparity map
    depth_map = np.zeros((calib_params['Height'], calib_params['Width']), dtype=np.float32) # depth map
    for pixel_int, depth_pixel in zip(uv_int, depth_uv):
        dis_map[pixel_int[1], pixel_int[0]] = calib_params["B"] / 1000 * calib_params["K"][0, 0] / depth_pixel   # depth --> disparity
        depth_map[pixel_int[1], pixel_int[0]] = depth_pixel

    return dis_map, depth_map


def show_clouds_with_color(image, depth_map, calib_params):
    # Recovering the colorized point cloud using Open3D.
    image_o3d = o3d.geometry.Image(image)
    depth_o3d = o3d.geometry.Image(depth_map)

    rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False, depth_scale=1.0, depth_trunc=20)

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.intrinsic_matrix = calib_params["K"]
    camera_intrinsic.height = calib_params['Height']
    camera_intrinsic.width = calib_params['Width']
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_o3d, camera_intrinsic)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()


if __name__ == "__main__":
    calib_params = read_calib_params('I:/data/dev_kit/calibration/calib_20230406.pkl')
    image_path = 'XXX/train/2023-04-08-02-33-11/left/20230408023909.400.jpg'
    pcd_path = 'XXX/train/2023-04-08-02-33-11/pcd/20230408023909.400.pcd'
    image = cv2.imread(image_path)
    cloud = o3d.io.read_point_cloud(pcd_path)
    cloud = np.asarray(cloud.points)

    uv, depth_uv = project_point2camera(calib_params, cloud)
    dis_map, depth_map = get_dis_depth_map(uv, depth_uv, calib_params)

    show_image_with_points(uv, depth_uv, image)
    show_clouds_with_color(image, depth_map, calib_params)
