# photoneo_data/20181217
p_robot=[[0.35705498188046264, -0.61592791298102, 0.142993429634809, -0.08591892905881739, -0.3886222774511321, 0.9151340190752171, 0.06419026584554777],
    [0.2990568677145358, -0.4899298457150618, 0.16996147198289374, -0.08595957781684653, -0.3886096368514568, 0.9151259847286154, 0.06432676893961795],
    [0.18206887844554992, -0.5469999549854314, 0.1659302750654687, -0.08646242706426885, -0.3889325826192631, 0.914938690401202, 0.06436526773190819],
    [0.26694493167509775, -0.5989386340792358, 0.1862799762422314, -0.08586592309673075, -0.3885202949278296, 0.9151261079205532, 0.06498638535970076],
    [0.27704067625331485, -0.573055166920657, 0.26205388882758757, -0.08580920098798522, -0.3893105864494028, 0.9148593368686363, 0.06408152657751885]]

# Accuracy test with ICP resgistration
# target = H * source, base_p = H * cam_p
from open3d import *
import numpy as np
import copy, tf
from matplotlib import pyplot as plt

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def evaluate_calibration(s,t,H):
    sTt = copy.deepcopy(s)
    sTt.transform(H)
    HI = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    reg_p2p = registration_icp(sTt, t, 0.0003, HI, TransformationEstimationPointToPoint(),ICPConvergenceCriteria(max_iteration = 2000))
    R = reg_p2p.transformation[:3,:3]
    T = reg_p2p.transformation[:3,3]
    al, be, ga = tf.transformations.euler_from_matrix(R, 'sxyz')
    print("xyz= [%2.2f, %2.2f, %2.2f]"%(T[0]*1000,T[1]*1000,T[2]*1000))
    print("rpy= [%2.5f, %2.5f, %2.5f]"%(al,be,ga))
    return T*1000, np.array([al, be, ga])

target = read_point_cloud("/home/bionicdl/photoneo_data/20181217/aubo-i5-EndFlange_cropped_m.pcd")
H = np.load("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/H_i.npy")

for i in range(1,5):
    source = read_point_cloud("/home/bionicdl/photoneo_data/20181217/tool0_%s.pcd"%(i+1))
    H_offset = np.matrix([[-1,0,0,0],[0,1,0,0],[0,0,-1,-0.006],[0,0,0,1]])
    H_base_tool = tf.transformations.quaternion_matrix(p_robot[i][3:])
    H_base_tool[:3,3] = np.array(p_robot[i][:3])
    s = copy.deepcopy(source)
    t = copy.deepcopy(target)
    t.transform(H_offset)
    t.transform(H_base_tool)
    xyz_i, rpy_i = evaluate_calibration(s,t,H)
    draw_registration_result(s, t, H)
