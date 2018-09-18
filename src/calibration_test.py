for i in [4]:
    cloud = pcl.load_XYZRGB('waypoint%s.ply'%(i+1))
    # TODO: PassThrough Filter
    passthrough = cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 380
    axis_max = 435
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
    # Ransac circle segmentation
    seg = tool0.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_CIRCLE3D)
    max_distance = 0.2
    seg.set_distance_threshold(max_distance)
    seg.set_MaxIterations(50000)
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
    p_camera_test = coefficients[:3]

p_camera_test = np.array(p_camera_test)/1000
p_camera_test = np.matrix([0.01669563, -0.01615144, 0.41323917, 1])
np.matmul(H, p_camera_test.T)
