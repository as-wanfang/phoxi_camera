import pcl

cloud = pcl.load_XYZRGB('Aubo_tool0_exposure10.ply')

# TODO: PassThrough Filter
passthrough = cloud.make_passthrough_filter()
filter_axis = 'z'
passthrough.set_filter_field_name (filter_axis)
axis_min = 0.4
axis_max = 0.435
passthrough.set_filter_limits (axis_min, axis_max)
cloud_filtered = passthrough.filter()

# TODO: Statistical outlier filter
outlier_filter = cloud.make_statistical_outlier_filter()
outlier_filter.set_mean_k(50)
outlier_filter.set_std_dev_mul_thresh(1.0)
cloud_filtered = outlier_filter.filter()

# TODO: RANSAC Plane Segmentation
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.01
seg.set_distance_threshold(max_distance)
inliers, coefficients = seg.segment()

# TODO: Extract inliers and outliers
cloud_table = cloud_filtered.extract(inliers, negative=False)
cloud_objects = cloud_filtered.extract(inliers, negative=True)
