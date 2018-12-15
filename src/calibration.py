import pcl, tf
import numpy as np
from sensor_stick.pcl_helper import *

# 20181205
p_robot = np.array([[0.3043, -0.5347, 0.1779, 1],
               [0.3724, -0.5580, 0.1992, 1],
               [0.3724, -0.5989, 0.1742, 1],
               [0.2899, -0.5989, 0.1742, 1]])
p_robot = p_robot.transpose()

# segment the flange, this step is very fast
for i in [1]:
    cloud = pcl.load('waypoint%s.ply'%(i+1))
    # TODO: PassThrough Filter
    passthrough = cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 320
    axis_max = 355
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
    ec.set_ClusterTolerance(0.5)    # Set tolerances for distance threshold 
    ec.set_MinClusterSize(2000)
    ec.set_MaxClusterSize(100000)   # as well as minimum and maximum cluster size (in points)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    tool_index = []
    # select the tool0 plane has the largest number of points
    for cluster in cluster_indices:
        if len(cluster)>len(tool_index):
            tool_index = cluster
    tool0 = white_cloud.extract(tool_index)
    pcl.save(tool0, "tool0_%s.pcd"%(i+1))

p_camera = []
for i in range(4):
    tool0 = pcl.load('tool0_%s.pcd'%(i+1))
    #proj = tool0.make_ProjectInliers()
    #proj.set_model_type(pcl.SACMODEL_PLANE)
    #tool0_ = proj.filter()
    #pcl.save(tool0_, "tool0_j%s.pcd"%(i+1))
    seg = tool0.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_CIRCLE3D)
    max_distance = 0.5
    seg.set_distance_threshold(max_distance)
    seg.set_MaxIterations(10000)
    seg.set_optimize_coefficients("true")
    seg.set_method_type(6)
    inliers, coefficients = seg.segment()
    clc = tool0.extract(inliers, negative=False)
    outliers = tool0.extract(inliers, negative=True)
    points_list = []
    for data in clc:
        points_list.append([data[0], data[1], data[2], rgb_to_float([0,255,0])])
    for data in outliers:
        points_list.append([data[0], data[1], data[2], rgb_to_float([255,0,0])])
    tool0_c = pcl.PointCloud_PointXYZRGB()
    tool0_c.from_list(points_list)
    pcl.save(tool0_c, "tool0_c%s.pcd"%(i+1))
    p_camera.append(coefficients[:3])
    coefficients[3]


p_camera_ = np.matrix(np.concatenate((np.array(p_camera).transpose()/1000, np.ones([1,4])),axis=0))
p_camera_reverse = p_camera_.getI()

H = np.matmul(p_robot, p_camera_reverse)

R = H[:3,:3]
t = H[:3, 3]

# calculate the rpy of rotation for usage in urdf
al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')

[16.695629119873047, -16.15144157409668, 413.2391662597656, 29.98028564453125, -0.015968363732099533, 0.019328217953443527, -0.9996856451034546]
