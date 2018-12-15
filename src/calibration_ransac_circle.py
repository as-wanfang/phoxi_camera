import pcl, tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sensor_stick.pcl_helper import *

p_robot = np.array([[0.3043, -0.5347, 0.1779, 1],
               [0.3724, -0.5580, 0.1992, 1],
               [0.3724, -0.5989, 0.1742, 1],
               [0.2899, -0.5989, 0.1742, 1]])
p_robot = p_robot.transpose()

# segment the flange, this step is very fast
for i in range(4):
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

# 3D Circle fittning using ransac and flange radius information
# unit of the pointcloud is mm
maxD = 0.3
R_FLANGE = 31
p_camera = []
for i in range(4):
    tool0 = pcl.load('tool0_%s.pcd'%(i+1))
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
        if (R_FLANGE-R)>1 or R_FLANGE-R<0:
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
    pcl.save(tool0_c, "Tool0_c%s.pcd"%(i+1))
    p_camera.append(max_P)
    max_R

p_camera_ = np.matrix(np.concatenate((np.array(p_camera).transpose()/1000, np.ones([1,4])),axis=0))
p_camera_reverse = p_camera_.getI()

H = np.matmul(p_robot, p_camera_reverse)

R = H[:3,:3]
t = H[:3, 3]

# calculate the rpy of rotation for usage in urdf
al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')

print("xyz= %s"%(t.getT()))
print("rpy= %s %s %s"%(al,be,ga))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2], s=0.5,  c='y', marker='.')
# ax.scatter(max_inliers[:,0], max_inliers[:,1], max_inliers[:,2],  s=1, c='r', marker='.')
# plt.show()
