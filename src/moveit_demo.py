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
from mpl_toolkits.mplot3d import Axes3D

NUM_ATTEMPS = 1
PLANNER_NAME = "RRTConnect"

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial',
                  anonymous=True)

robot = moveit_commander.RobotCommander()

scene = moveit_commander.PlanningSceneInterface()

group = moveit_commander.MoveGroupCommander("Arm")


display_trajectory_publisher = rospy.Publisher(
                                      '/move_group/display_planned_path',
                                 moveit_msgs.msg.DisplayTrajectory,
                                      queue_size=20)

# group.get_current_joint_values()
# above bin picking
#pick_joint_positions = [-1.5534956288425719, -1.6570561804545578, 0.9666828931631055, -0.8975538597476668, -1.588098231510399, 9.098554519005121e-05]
# deep picking
pick_joint_positions = [-1.6571229807091412, -1.8987741030128207, 1.8297206708493179, -1.4844425337884575, -1.588071931215888, -0.10369734183303081]
place_joint_positions = [4.72094163298607e-05, -1.2083540076804813, 1.3808265007842797, -1.7260865345759318, -1.5880401480919681, -9.419479356147349e-05]

# specify planner

group.set_planner_id(PLANNER_NAME+'kConfig1')
group.set_num_planning_attempts(1)
group.set_planning_time(5)

plans_pick = []
plans_place = []
for i in range(2):
  group.set_start_state_to_current_state()
  group.clear_pose_targets()
  if abs(group.get_current_joint_values()[0]-pick_joint_positions[0])<0.001:
    group.set_joint_value_target(place_joint_positions)
    for j in range(8):
      plan = group.plan()
      if len(plan.joint_trajectory.points)==0:
        continue
      plans_place.append(plan)
      rospy.sleep(0.5)
    if len(plans_place)==0:
      print('Fail to get any place plan!')
      break
    group.execute(plans_place[0])
    rospy.sleep(3)
  else:
    group.set_joint_value_target(pick_joint_positions)
    for j in range(8):
      plan = group.plan()
      if len(plan.joint_trajectory.points)==0:
        continue
      plans_pick.append(plan)
      rospy.sleep(0.5)
    if len(plans_pick)==0:
      print('Fail to get any pick plan!')
      break
    group.execute(plans_pick[0])
    rospy.sleep(3)

print('sucess rate for pick: %s and place: %s'%(len(plans_pick),len(plans_place)))
#bag = rosbag.Bag('test.bag', 'w')
#bag.write('plan',plan1)
#bag.close()

# JointTrajectoryPoint
#float64[] positions, a list of joint values for each joint, velocities, accelerations, effort, duration time_from_start
######################################################################
# # calculate end effector length
plans = plans_place
DIR = '/home/wanfang/moveit_results/place/'

moveit_fk = rospy.ServiceProxy('compute_fk', moveit_msgs.srv.GetPositionFK)
joint_names = plans[0].joint_trajectory.joint_names
header = plans[0].joint_trajectory.header
fkln = ['wrist_3_link']
# rosservice info /compute_fk
rs = moveit_msgs.msg.RobotState()
rs.joint_state.name = joint_names

paths_ee_position = []
for plan in plans:
  path = []
  for point in plan.joint_trajectory.points:
    rs.joint_state.position = point.positions
    position = moveit_fk(header, fkln, rs).pose_stamped[0].pose.position  #.position .orientation
    path.append([position.x, position.y, position.z])
  paths_ee_position.append(path)

paths_length = []
for path in paths_ee_position:
  length = 0
  for i in range(1,len(path)):
    length = length + np.sqrt( sum( (np.array(path[i]) - np.array(path[i-1]))**2 ) )
  paths_length.append(length)

paths_position = [np.array([list(point.positions) for point in plan.joint_trajectory.points]) for plan in plans]
paths_velocity = [np.array([list(point.velocities) for point in plan.joint_trajectory.points]) for plan in plans]
paths_velocity = [np.array([list(point.velocities) for point in plan.joint_trajectory.points]) for plan in plans]
paths_time = [np.array([point.time_from_start.secs+point.time_from_start.nsecs/1000000000.0 for point in plan.joint_trajectory.points]) for plan in plans]

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax = ax.ravel()
fig.set_size_inches(18.5, 10.5, forward=True)
for j in range(16):
  ax[j].set_title('joint velocity plan %s'%(j+1))
  for i in range(6):
    ax[j].plot(paths_time[j], paths_velocity[j][:,i],label="%d"%(i+1))

plt.xlabel('time (s)')
leg = plt.legend(loc='best')
leg.get_frame().set_alpha(0.5)
fig.savefig(DIR+'%s_%s_joint_velocity.png'%(PLANNER_NAME,NUM_ATTEMPS), bbox_inches='tight')

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax = ax.ravel()
fig.set_size_inches(18.5, 10.5, forward=True)
for j in range(16):
  ax[j].set_title('joint position plan %s'%(j+1))
  for i in range(6):
    ax[j].plot(paths_time[j], paths_position[j][:,i],label="%d"%(i+1))

plt.xlabel('time (s)')
leg = plt.legend(loc='best')
leg.get_frame().set_alpha(0.5)
fig.savefig(DIR+'%s_%s_joint_position.png'%(PLANNER_NAME,NUM_ATTEMPS), bbox_inches='tight')

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax = ax.ravel()
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].plot(paths_length)
ax[0].set_title('path lengths for %s plans'%(len(paths_length)))
ax[1].plot([t_serial[-1] for t_serial in paths_time])
ax[1].set_title('path time cost for %s plans'%(len(paths_length)))
fig.savefig(DIR+'%s_%s_path_stat.png'%(PLANNER_NAME,NUM_ATTEMPS), bbox_inches='tight')
#####################################################
# test pick
# group.set_start_state_to_current_state()
# group.set_joint_value_target(pick_joint_positions)
# valid_count = 0
# for k in range(20):
#   plan = group.plan()
#   if len(plan.joint_trajectory.points)==0:
#     continue
#   paths_position = [np.array([list(point.positions) for point in plan.joint_trajectory.points])]
#   for i in range(6):
#     plt.plot(paths_position[0][:,i])
#   plt.show()
#   valid_count += 1
