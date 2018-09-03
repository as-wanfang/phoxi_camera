#!/usr/bin/env python

import rospy
from phoxi_camera.srv import *
from std_srvs.srv import *
import perception as p

def depth_im_callback(self, msg):
    """Callback for handling depth images.
    """
    cur_depth_im = DepthImage(self._bridge.imgmsg_to_cv2(msg) / 1000.0, frame=self._frame)
    print "Get current depth image!"

if __name__ == '__main__':
    
    # initialize the ROS node
    rospy.init_node('Photoneo_Control_Node')

    sensor = p.PhoXiSensor(device_name='1711004')
    #sensor.start()
    success = rospy.ServiceProxy('phoxi_camera/connect_camera', ConnectCamera)('1711004').success
    if not success:
        print "Could not connect to PhoXi sensor!"
    depth_im_sub = rospy.Subscriber('/phoxi_camera/depth_map', ImageMessage, depth_im_callback)
    rospy.loginfo('Sensor Running')
    
    # get the images from the sensor
    #color_image, depth_image, _ = sensor.frames()

    rospy.ServiceProxy('phoxi_camera/start_acquisition', Empty)()
    rospy.ServiceProxy('phoxi_camera/trigger_image', TriggerImage)()

        self._cur_color_im = None
        self._cur_depth_im = None
        self._cur_normal_map = None

        rospy.ServiceProxy('phoxi_camera/get_frame', GetFrame)(-1)
        print "Finish get_frame!"
    print "frames:"
    #color_image

    rospy.spin()

