 # Test calibration accuracy with rviz
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import moveit_msgs.srv
import geometry_msgs.msg
from std_msgs.msg import String
import numpy as np

# NUM_ATTEMPS = 1
# PLANNER_NAME = "RRTConnect"
# moveit_commander.roscpp_initialize(sys.argv)
# rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
# robot = moveit_commander.RobotCommander()
# scene = moveit_commander.PlanningSceneInterface()
# group = moveit_commander.MoveGroupCommander("manipulator_i5")

# photoneo_data/20181217
p1=[0.35705498188046264, -0.61592791298102, 0.142993429634809+0.022, -0.08591892905881739, -0.3886222774511321, 0.9151340190752171, 0.06419026584554777]
p2=[0.2990568677145358, -0.4899298457150618, 0.16996147198289374+0.022, -0.08595957781684653, -0.3886096368514568, 0.9151259847286154, 0.06432676893961795]
p3=[0.18206887844554992, -0.5469999549854314, 0.1659302750654687+0.022, -0.08646242706426885, -0.3889325826192631, 0.914938690401202, 0.06436526773190819]
p4=[0.26694493167509775, -0.5989386340792358, 0.1862799762422314+0.022, -0.08586592309673075, -0.3885202949278296, 0.9151261079205532, 0.06498638535970076]
p5=[0.27704067625331485, -0.573055166920657, 0.26205388882758757+0.022, -0.08580920098798522, -0.3893105864494028, 0.9148593368686363, 0.06408152657751885]
# group.set_pose_targets([p5])
# plan = group.plan()
# group.execute(plan)

# Accuracy test with ICP resgistration
# target = H * source, base_p = H * cam_p
from open3d import *
import numpy as np
import copy, tf
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

target = read_point_cloud("aubo-i5-EndFlange_cropped_m.pcd")
source = read_point_cloud("tool0_5.pcd")
H_offset = np.matrix([[-1,0,0,0],[0,1,0,0],[0,0,-1,-0.006],[0,0,0,1]])
H_base_tool = tf.transformations.quaternion_matrix([-0.08580920098798522, -0.3893105864494028, 0.9148593368686363, 0.06408152657751885])
H_base_tool[:3,3] = np.array([0.27704067625331485, -0.573055166920657, 0.26205388882758757])
s = copy.deepcopy(source)
t = copy.deepcopy(target)
t.transform(H_offset)
t.transform(H_base_tool)

H = np.load("H.npy")
H1 = np.load("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/H1.npy")
H2 = np.load("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/H_ransac.npy")
H3 = np.load("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/H_s.npy")

# show transform from cam_p to world_p
# draw_registration_result(s, t, H)

# quantify calibration error
sTt = copy.deepcopy(s)
sTt.transform(H1)
HI = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
reg_p2p = registration_icp(sTt, t, 0.0003, HI, TransformationEstimationPointToPoint(),ICPConvergenceCriteria(max_iteration = 2000))
R = reg_p2p.transformation[:3,:3]
T = reg_p2p.transformation[:3,3]
al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')
print("xyz= [%2.2f, %2.2f, %2.2f]"%(T[0]*1000,T[1]*1000,T[2]*1000))
print("rpy= [%2.5f, %2.5f, %2.5f]"%(al,be,ga))
# results
# H:        xyz= [0.00040029 0.00081483 0.00168168] rpy= 0.00333746699586 -0.00112373840469 -0.000103912228989
# H1:       xyz= [0.00054628 0.00065536 0.00126683] rpy= 0.00235151356392 -0.000893351140693 -0.000616103236213
# H_ransac: xyz= [-0.00013903  0.00012276  0.00059244] rpy= 0.000806734526165 0.000140151745263 0.000170308052559

# quantify icp
threshold = 0.0003
evaluation = evaluate_registration(sTt, t, threshold, reg_p2p.transformation)
print(evaluation)
draw_registration_result(sTt, t, reg_p2p.transformation)
# results
# H1: RegistrationResult with fitness = 0.998582, inlier_rmse = 0.000396, and correspondence_set size of 23244
# H_ransac: RegistrationResult with fitness = 0.998754, inlier_rmse = 0.000352, and correspondence_set size of 23248

estimate_normals(s, search_param = KDTreeSearchParamHybrid(radius = 0.01, max_nn = 30))
estimate_normals(t, search_param = KDTreeSearchParamHybrid(radius = 0.01, max_nn = 30))
reg_p2l = registration_icp(s, t, threshold, HI, TransformationEstimationPointToPlane())
R = reg_p2l.transformation[:3,:3]
T = reg_p2l.transformation[:3,3]
al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')
print("xyz= %s"%t)
print("rpy= %s %s %s"%(al,be,ga))
evaluation = evaluate_registration(s, t, threshold, reg_p2l.transformation)

##############################################################################################################
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
