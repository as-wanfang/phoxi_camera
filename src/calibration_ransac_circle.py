import pcl, tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sensor_stick.pcl_helper import *

# p_robot = np.array([[0.3043, -0.5347, 0.1779, 1],
#                [0.3724, -0.5580, 0.1992, 1],
#                [0.3724, -0.5989, 0.1742, 1],
#                [0.2899, -0.5989, 0.1742, 1]])

# photoneo_data/20181217
p_robot = np.array([[0.35705498188046264, -0.61592791298102, 0.15799342963480897, 1],
                    [0.2990568677145358, -0.4899298457150618, 0.16996147198289374, 1],
                    [0.18206887844554992, -0.5469999549854314, 0.1659302750654687, 1],
                    [0.26694493167509775, -0.5989386340792358, 0.1862799762422314, 1]
])

p_robot[:,2] = p_robot[:,2] - 0.015

p_robot = p_robot.transpose()

# segment the flange, this step is very fast
for i in range(5):
    cloud = pcl.load('/home/bionicdl/photoneo_data/20181217/waypoint%s.ply'%(i+1))
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
    pcl.save(flange, "tool0_%s.pcd"%(i+1))

# 3D Circle fittning using ransac and flange radius information
# unit of the pointcloud is mm
maxD = 0.3/1000
R_FLANGE = 31.0/1000
detR = 0.001 #
p_camera = []
for i in range(5):
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
    pcl.save(tool0_c, "Tool0_c%s.pcd"%(i+1))
    p_camera.append(max_P)
    max_R
    max_P

p_camera_ = np.matrix(np.concatenate((np.array(p_camera).transpose()/1, np.ones([1,5])),axis=0))
np.save("p_camera.npy",p_camera_)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2], s=0.5,  c='y', marker='.')
# ax.scatter(max_inliers[:,0], max_inliers[:,1], max_inliers[:,2],  s=1, c='r', marker='.')
# plt.show()

####################################################
# Direct matrix multiplication
p_camera_reverse = p_camera_[:,:4].getI()
H = np.matmul(p_robot, p_camera_reverse)
R = H[:3,:3]
t = H[:3, 3]
# calculate the rpy of rotation for usage in urdf
al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')
print("xyz= %s"%(t.getT()))
print("rpy= %s %s %s"%(al,be,ga))

#########################################################
# calculate R using SVD
p_robot_centroid = np.mean(p_robot[:3,:],axis=1).reshape(3,1)
p_camera_centroid = np.mean(p_camera_[:3,:4],axis=1).reshape(3,1)

p_robot_demean = p_robot[:3,:] - np.tile(p_robot_centroid,[1,4])
p_camera_demean = p_camera_[:3,:4] - np.tile(p_camera_centroid,[1,4])

r = np.matrix(np.zeros([3,3]))
for i in range(4):
    r += np.matmul(p_camera_demean[:,i].reshape(3,1),p_robot_demean[:,i].reshape(1,3))


u, s, vt = np.linalg.svd(r)
R = np.matmul(vt.transpose(), u.transpose())

t = p_robot_centroid - np.matmul(R, p_camera_centroid)

al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')

print("xyz= %s"%(t.getT()))
print("rpy= %s %s %s"%(al,be,ga))

H = np.zeros([4,4])
H[:3,:3]  = R
H[:3, 3] = t.transpose()
H[3,3] = 1
