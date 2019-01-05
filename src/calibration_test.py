 # Test calibration accuracy with rviz
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import moveit_msgs.srv
import geometry_msgs.msg
from std_msgs.msg import String
import rosbag
import numpy as np
from matplotlib import pyplot as plt

NUM_ATTEMPS = 1
PLANNER_NAME = "RRTConnect"

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial',
                  anonymous=True)

robot = moveit_commander.RobotCommander()

scene = moveit_commander.PlanningSceneInterface()

group = moveit_commander.MoveGroupCommander("manipulator_i5")

# photoneo_data/20181217
p1=[0.35705498188046264, -0.61592791298102, 0.142993429634809+0.022, -0.08591892905881739, -0.3886222774511321, 0.9151340190752171, 0.06419026584554777]
p5=[0.27704067625331485, -0.573055166920657, 0.26205388882758757-0.015+0.022, -0.08580920098798522, -0.3893105864494028, 0.9148593368686363, 0.06408152657751885]
group.set_pose_targets([p5])
plan = group.plan()
group.execute(plan)

# Accuracy test with ICP resgistration
from open3d import *
import numpy as np
import copy, tf
H = np.load("H.npy")
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

source = read_point_cloud("flange_aubo-i5_m.pcd")
target = read_point_cloud("tool0_5.pcd")
threshold = 0.02

H_offset = np.matrix([[-1,0,0,0],[0,1,0,0],[0,0,-1,-0.006],[0,0,0,1]])
H_base_tool = tf.transformations.quaternion_matrix([-0.08580920098798522, -0.3893105864494028, 0.9148593368686363, 0.06408152657751885])
H_base_tool[:3,3] = np.array([0.27704067625331485, -0.573055166920657, 0.26205388882758757 - 0.015])

s = copy.deepcopy(source)
t = copy.deepcopy(target)
s.transform(H_offset)
s.transform(H_base_tool)

draw_registration_result(t, s, H)

evaluation = evaluate_registration(s, t, threshold, H)
print(evaluation)

reg_p2p = registration_icp(s, t, threshold, H,
            TransformationEstimationPointToPoint())

draw_registration_result(source, target, reg_p2p.transformation)

print("Apply point-to-plane ICP")
reg_p2l = registration_icp(source, target, threshold, trans_init,
            TransformationEstimationPointToPlane())
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
print("")
draw_registration_result(source, target, reg_p2l.transformation)

# Test calibration accuracy with nail
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
