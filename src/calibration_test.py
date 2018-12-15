for i in [4]:
    cloud = pcl.load_XYZRGB('destination.ply')
    # TODO: PassThrough Filter
    passthrough = cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 558
    axis_max = 570
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
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(10000)   # as well as minimum and maximum cluster size (in points)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    tool_index = []
    # select the tool0 plane has the largest number of points
    for cluster in cluster_indices:
        if len(cluster)>len(tool_index):
            tool_index = cluster
    nail = white_cloud.extract(tool_index)
    pcl.save(nail, "nail.pcd")
    p_camera_test = np.mean(nail.to_array(), axis=0)


p_camera_test = np.array(p_camera_test)/1000
p_camera_test = np.matrix([p_camera_test[0], p_camera_test[1], p_camera_test[2], 1])
np.matmul(H, p_camera_test.T)
