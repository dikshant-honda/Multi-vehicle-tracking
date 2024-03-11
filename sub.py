#!/usr/bin/env python3

import rospy
# from std_msgs.msg import String
from multi_vehicle_tracking.msg import pos_and_vel

def callback(data):
    print(data.id)
    
def listener():
    rospy.init_node('test', anonymous=True)
    rospy.Subscriber("pos_and_vel_data", pos_and_vel, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()