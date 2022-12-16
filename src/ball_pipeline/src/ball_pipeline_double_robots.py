#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray
import tf
from visualization_msgs.msg import Marker
from apriltag_ros.msg import AprilTagDetectionArray

RVIZ_VIS = True


class tf_broadcaster:

    def __init__(self):
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.depth_image = None
        self.intrinsics = None
        self.ball_pos = None
        self.marker = Marker()
        self.marker.header.frame_id = "/map"
        self.init = 1
        self.init_t265 = 1
        self.robot_pos = [0, 0, 0, 0, 0, 0, 1]

        # self.ball_pos_sub = message_filters.Subscriber("/camera_ball_pos",Int16MultiArray)
        # self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",Image)
        self.ball_pos_sub = rospy.Subscriber('/camera_ball_pos', Float32MultiArray, self.tf_callback)
        self.shooting_ball_pos_pub = rospy.Publisher("/ball_world", Float32MultiArray, queue_size=1)
        self.ball_pos_pub = rospy.Publisher("/global_ball_pos", Float32MultiArray, queue_size=1)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=2)
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.ball_pos_sub, self.depth_sub], 10, 0.1, allow_headerless=True)
        # self.ts.registerCallback(self.image_callback)
        self.robot_pos_pub = rospy.Publisher("/global_robot_pos", Float32MultiArray, queue_size=1)
        self.tag_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.tag_callback)

    def tf_callback(self, data):
        print("receved ball position")
        self.ball_pos = data.data
        # print(self.ball_pos)
        self.br.sendTransform((self.ball_pos[0], self.ball_pos[1], self.ball_pos[2]),
                              tf.transformations.quaternion_from_euler(0, 0, 0),
                              rospy.Time.now(),
                              "/ball",
                              "/camera_ext")  # 843112070756 lower
        # 843112072265 upper - this is camera_ext
        if self.init:
            self.listener.waitForTransform('/map', '/ball', rospy.Time(), rospy.Duration(4.0))
        else:
            self.init = 0
        (ball_init_pos, ball_init_rot) = self.listener.lookupTransform('/map', '/ball', rospy.Time(0))
        msg = Float32MultiArray()
        msg.data = [ball_init_pos[0], ball_init_pos[1], ball_init_pos[2]]
        print("ball", ball_init_pos)
        self.ball_pos_pub.publish(msg)

        self.br.sendTransform((self.ball_pos[0], self.ball_pos[1], self.ball_pos[2]),
                              tf.transformations.quaternion_from_euler(0, 0, 0),
                              rospy.Time.now(),
                              "/shooting_ball",
                              "/camera_color_optical_frame")  # 843112070756 lower

        self.listener.waitForTransform('/world', '/camera_color_optical_frame', rospy.Time(), rospy.Duration(1.0))
        (shooting_ball_pos, shooting_ball_rot) = self.listener.lookupTransform('/world', '/shooting_ball',
                                                                               rospy.Time(0))
        shooting_msg = Float32MultiArray()
        shooting_msg.data = [shooting_ball_pos[0], shooting_ball_pos[1], shooting_ball_pos[2]]
        print("shooting ball", shooting_ball_pos)

        self.shooting_ball_pos_pub.publish(shooting_msg)

        if RVIZ_VIS:
            self.marker.header.stamp = rospy.Time.now()
            # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            self.marker.type = 2
            self.marker.id = 0

            # Set the scale of the self.marker
            self.marker.scale.x = 0.2
            self.marker.scale.y = 0.2
            self.marker.scale.z = 0.2

            # Set the color
            self.marker.color.r = 1.0
            self.marker.color.g = 0.0
            self.marker.color.b = 0.0
            self.marker.color.a = 1.0

            # Set the pose of the self.marker
            self.marker.pose.position.x = ball_init_pos[0]
            self.marker.pose.position.y = ball_init_pos[1]
            self.marker.pose.position.z = ball_init_pos[2]
            self.marker.pose.orientation.x = 0.0
            self.marker.pose.orientation.y = 0.0
            self.marker.pose.orientation.z = 0.0
            self.marker.pose.orientation.w = 1.0
            self.marker_pub.publish(self.marker)

    def publish_robot_pos(self):

        if self.init_t265:
            try:
                (trans, rot) = self.listener.lookupTransform('/t265_odom_frame', '/dog', rospy.Time(0))
                print("T265 Initial Pos: ", trans, "\n\n\n")
                trans[2] = trans[2] - 0.21
                self.br.sendTransform(trans,
                                      rot,
                                      rospy.Time.now(),
                                      "/map",
                                      "/t265_odom_frame")
                self.listener.waitForTransform('/map', '/dog', rospy.Time(), rospy.Duration(4.0))
                self.init_t265 = 0
                self.robot_init_pos = (trans, rot)
            except Exception as e:
                pass

        else:
            self.br.sendTransform(self.robot_init_pos[0],
                                  self.robot_init_pos[1],
                                  rospy.Time.now(),
                                  "/map",
                                  "/t265_odom_frame")
            try:
                (robot_pos, robot_rot) = self.listener.lookupTransform('/map', '/dog', rospy.Time(0))
                msg = Float32MultiArray()
                msg.data = [robot_pos[0], robot_pos[1], robot_pos[2], robot_rot[0], robot_rot[1], robot_rot[2],
                            robot_rot[3]]
                print("robot", robot_pos)
                self.robot_pos_pub.publish(msg)
            except Exception as e:
                print(e)

    def tag_callback(self, tag_msg):
        # print(tag_msg.detections[0].id[0])
        # global goal_cam
        # print("called")
        # for i in tag_msg.detections:
        #     if i.id[0] == 4:
        #       self.robot_pos = [-i.pose.pose.pose.position.x, -i.pose.pose.pose.position.y,
        #                   -i.pose.pose.pose.position.z, i.pose.pose.pose.orientation.x, i.pose.pose.pose.orientation.y,
        #                   i.pose.pose.pose.orientation.z, i.pose.pose.pose.orientation.w]
        # self.br.sendTransform(tuple(self.robot_pos[:3]),
        #                tuple(self.robot_pos[3:]),
        #                rospy.Time.now(),
        #                "/camera_ext",
        #                "/dog") #843112070756 lower
        #                #843112072265 upper - this is camera_ext
        if self.init:
            self.listener.waitForTransform('/map', '/dog', rospy.Time(), rospy.Duration(4.0))
        else:
            self.init = 0
        (ball_init_pos, ball_init_rot) = self.listener.lookupTransform('/map', '/dog', rospy.Time(0))
        msg = Float32MultiArray()
        msg.data = [ball_init_pos[0], ball_init_pos[1], ball_init_pos[2], ball_init_rot[0], ball_init_rot[1],
                    ball_init_rot[2], ball_init_rot[3]]
        print("robot", ball_init_pos)
        self.robot_pos_pub.publish(msg)


def main(args):
    rospy.init_node('tf_broadcast', anonymous=True)
    ic = tf_broadcaster()
    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():
        # ic.publish_robot_pos()
        # rate.sleep()
        rospy.spin()

    # except KeyboardInterrupt:
    #   print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)