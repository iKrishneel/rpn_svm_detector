#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2 as cv
import rospy
import time

from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32

from detector import RPNSVMDetector
from rpn_svm_detector.srv import *

class RSDetector(RPNSVMDetector):
    def __init__(self):
        
        mrcnn_model = rospy.get_param('~mrcnn_model', None)
        # self.__model_path = rospy.get_param('~model_file', None)
        self.__model_path = None
        self.__is_service = rospy.get_param('~is_service', False)
        self.__debug = rospy.get_param('~debug', True)

        mrcnn_model = '../models/mask_rcnn_coco.h5'
        super(RSDetector, self).__init__(mrcnn_model, is_train=False)
        
        if not self.__is_service:
            self.pub_detection = rospy.Publisher('/zsd_detector/bbox', PolygonStamped, queue_size=1)

        self.__object_name = None
        self.__loaded_model = True
        self.__exec_annotation_flag = False
        self.__exec_training_flag = False
        self.__exec_detection_flag = False


    def load_model(self):
        text_fn = os.path.join(os.environ['HOME'], 'Desktop/model_file.txt')
        if not os.path.isfile(self.__text_fn):
            rospy.loginfo('MODEL DOES NOT EXIST')
            return
        else:
            lines = [line.rstrip('\n')
                     for line in open(self.__text_fn)
            ]
            if len(lines) is 0:
                rospy.loginfo('INVALID NAME AT FILE LOCATION. UNKNOW OBJECT')
                self.__object_name = None
            else:
                self.__object_name = lines[-1]
                self.__model_path = lines[0]

                if os.path.isfile(self.__model_path):
                    rospy.loginfo('LOADING TRAINED MODEL')
                    self.load_trained_model
 
        
    def callback(self, image_msg):
        image = self.convert_to_cv_image(image_msg)
        if not image is None:
            polygon = self.run_zsd(image, image_msg.header)
            polygon.header = image_msg.header
            
            if not self.__is_service:
                self.pub_detection.publish(polygon)
            else:
                return ZeroShotDetectorResponse(polygon)
        else:
            rospy.logwarn('Empty image')
                                     
        
    def run_zsd(self, image):
        """
        run the trained detector
        """
        if self.__loaded_model:
            if self.load_trained_model(self.__model_path):
                self.__loaded_model = False
            else:
                rospy.logerr('TRAINED MODEL CANNOT BE LOADED FROM: {}'.format(self.__model_path))
                return
        
        polygon = self.detect(image)

        if polygon is None:
            rospy.logwarn('NO OBJECT DETECTED')
            return

        polygon = PolygonStamped()
        tl = Point32()
        tl.x = polygon[0]
        tl.y = polygon[1]

        br = Point32()
        br.x = polygon[2]
        br.y = polygon[3]
        polygon.points.append(tl)
        polygon.points.append(br)
        return polygon
        

    def collect_dataset(self):
        """
        function to collect the dataset
        """
        ## TODO
        pass

    def convert_to_cv_image(self, image_msg):
        """
        cv_bride for converting ros image to cv image
        """
        if image_msg is None:
            return None

        width = image_msg.width
        height = image_msg.height
        channels = int(len(image_msg.data) / (width * height))
        
        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1
            
        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)
            
        return cv_img

    def subscribe(self):
        rospy.Subscriber('image', Image, self.callback, queue_size=1)

    def service_handler(self, request):

        if request.exec_step.lower() == 'annotate':
            pass
        elif request.exec_step.lower() == 'train':
            pass
        elif request.exec_step.lower() == 'detect':
            if self.__object_name is None:
                self.__object_name = request.object_name

            return self.callback(request.image)

    def service(self):
        rospy.loginfo('SETTING UP SRV')
        srv = rospy.Service('zsd_detector', ZeroShotDetector, self.service_handler)    

def main(argv):
    try:
        rospy.init_node('zero_shot_detector')
        detector = RSDetector()
        rospy.spin()
    except rospy.ROSInitException as e:
        rospy.logerr(e)
        

if __name__ == "__main__":
    main(sys.argv)
