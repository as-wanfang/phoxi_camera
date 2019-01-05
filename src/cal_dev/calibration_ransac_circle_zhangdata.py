import pcl, tf
import numpy as np
from sensor_stick.pcl_helper import *
from functools import reduce
import time
from utils import *
from numpy.linalg import *
import os

data_path = "/home/bionicdl/photoneo_data/calibration_images/data_ransac10000"
index_list, p_camera_list, p_robot_list = read_data(data_path+"/data.txt", type="point_pair")
# p_robot_mat = np.array(p_robot_list).transpose()

for index in index_list:
    try:
        cloud = pcl.load(data_path+'/im%s.ply'%(index[:-1]))
    except:
        print("Not existed pointcloud!")
        continue
    # TODO: PassThrough Filter
    passthrough = cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.2
    axis_max = 0.5
    passthrough.set_filter_limits (axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    # TODO: Statistical outlier filter
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    outlier_filter.set_std_dev_mul_thresh(1.0)
    cloud_filtered = outlier_filter.filter()
    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_filtered)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.0005)    # Set tolerances for distance threshold
    ec.set_MinClusterSize(2000)
    ec.set_MaxClusterSize(100000)   # as well as minimum and maximum cluster size (in points)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    tool_index = []
    # select the tool0 plane has the largest number of points
    min_height = 10010
    current_length = 0
    for cluster in cluster_indices:
        cloud = white_cloud.extract(cluster)
        cloud_array = np.array(cloud)
        length = cloud_array.shape[0]
        height = cloud_array[:,2].min()
        if height < min_height:
            min_height = height
            tool_index = cluster
    tool0 = white_cloud.extract(tool_index)
    # plane segment
    seg = tool0.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    max_distance = 0.001
    seg.set_distance_threshold(max_distance)
    seg.set_MaxIterations(10000)
    seg.set_optimize_coefficients("true")
    seg.set_method_type(0)
    inliers, coefficients = seg.segment()
    flange = tool0.extract(inliers, negative=False)
    if len(inliers)>13000:
        pcl.save(flange, "/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_used/tool0_%s.pcd"%(index[:-1]))
        print("Flange saved!")

p_robot_used = []
maxD = 0.3/1000
R_FLANGE = 31.0/1000
detR = 0.001 #
p_camera = []
for i in range(len(index_list)):
    index = index_list[i]
    try:
        tool0 = pcl.load("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_used/tool0_%s.pcd"%(index[:-1]))
        # os.system("pcl_viewer /home/bionicdl/photoneo_data/calibration_images/data_ransac10000/im%s.PCD"%(index[:-1]))
    except:
        continue
    p_robot_used.append(p_robot_list[i])
    points = tool0.to_array()
    max_num_inliers = 0
    for k in range(10000):
        idx = np.random.randint(points.shape[0], size=3)
        A, B, C = points[idx,:]
        a = np.linalg.norm(C - B)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(B - A)
        s = (a + b + c) / 2
        R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
        if (R_FLANGE-R)>detR or R_FLANGE-R<0:
            continue
        b1 = a*a * (b*b + c*c - a*a)
        b2 = b*b * (a*a + c*c - b*b)
        b3 = c*c * (a*a + b*b - c*c)
        P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
        P /= b1 + b2 + b3
        num_inliers = 0
        inliers = []
        outliers = []
        for point in points:
            d = np.abs(np.linalg.norm(point-P)-R)
            if d < maxD:
                num_inliers += 1
                inliers.append(point)
            else:
                outliers.append(point)
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            max_A = A
            max_B = B
            max_C = C
            max_R = R
            max_P = P
            max_inliers = np.array(inliers)
            max_outliers = np.array(outliers)
    points_list = []
    for data in max_inliers:
        points_list.append([data[0], data[1], data[2], rgb_to_float([0,255,0])])
    for data in max_outliers:
        points_list.append([data[0], data[1], data[2], rgb_to_float([255,0,0])])
    tool0_c = pcl.PointCloud_PointXYZRGB()
    tool0_c.from_list(points_list)
    pcl.save(tool0_c, "/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_used/Tool0_c%s.pcd"%(i+1))
    p_camera.append(max_P)
    max_R
    max_P

p_robot_mat = np.array(p_robot_used).transpose()
p_camera_mat = np.array(p_camera).transpose()
np.save("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_used/p_robot_mat.npy",p_robot_mat)
np.save("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_used/p_camera_mat.npy",p_camera_mat)

#######################################################
# calibration
from calibration import get_calibrate
import numpy as np
import tf
p_robot_mat = np.load("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_used/p_robot_mat.npy")
p_camera_mat = np.load("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_used/p_camera_mat.npy")

calibrate = get_calibrate(4)
H = calibrate(p_robot_mat, p_camera_mat)
R = H[:3,:3]
t = H[:3,3]
al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')
print("xyz= %s"%(t.getT()))
print("rpy= %s %s %s"%(al,be,ga))

###################################################
# Ransac
def error_test(H):
    error_matrix = p_robot_mat - np.matmul(H_i[:3,:3],p_camera_mat) - np.tile(H_i[:3,3],[1,27])
    error = np.sum((np.asarray(error_matrix))**2)/p_camera_mat.shape[0]/(p_camera_mat.shape[1]-4)
    return error

min_error = 10010
for i in range(1):
    idx = np.random.choice(p_robot_mat.shape[1], 27,0)
    p_robot_mat_i = p_robot_mat[:,idx]
    p_camera_mat_i = p_camera_mat[:,idx]
    H_i = calibrate(p_robot_mat_i, p_camera_mat_i)
    error_matrix = p_robot_mat - np.matmul(H_i[:3,:3],p_camera_mat) - np.tile(H_i[:3,3],[1,27])
    error = np.mean( (np.sum((np.asarray(error_matrix))**2,axis=0))**(0.5) )
    # print("Id:{} Iteration:{}  Error:{} mm".format(idx, i, error*1000))
    if error < min_error:
        min_error = error
        H = H_i

# Ransac 4 points: min_error=0.520 with 1000 iterations
# All points: min_error = 0.517

p_robot_mat_i = p_robot_mat
p_camera_mat_i = p_camera_mat
for i in range(20):
    H_i = calibrate(p_robot_mat_i, p_camera_mat_i)
    error_matrix = p_robot_mat_i - np.matmul(H_i[:3,:3],p_camera_mat_i) - np.tile(H_i[:3,3],[1, p_camera_mat_i.shape[1]])
    error = (np.sum((np.asarray(error_matrix))**2,axis=0))**(0.5)
    print("Iteration:{}  Error:{} mm".format(i, np.mean(error*1000)))
    p_robot_mat_i = np.delete(p_robot_mat_i, np.argmax(error),1)
    p_camera_mat_i = np.delete(p_camera_mat_i, np.argmax(error),1)
