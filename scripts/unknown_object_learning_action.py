#!/usr/bin/env python

import os
import sys
import time
import message_filters
from cv_bridge import CvBridge

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

from ..zero_shot_svm_detector import ZeroShotSVMDetector

class UnknownObjectLearningAction(object):

    def __init__(self):
        
        zsd_service_name = rospy.get_param('zsd_service_name', 'zsd_detector')

        self.__bridge = CvBridge()
        self.__im_subdir = None
        self.__mk_subdir = None
        
        self.__save_signal = False

        self.__im_rgb = None
        self.__im_mask = None

        self.__text_fn = os.path.join(os.environ['HOME'], 'Desktop/model_file.txt')

        #! check if filename exist
        if not os.path.isfile(self.__text_fn):
            rospy.loginfo('NEW OBJECT')
            self.__object_name = None
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
                
        
        self.__lock = False
        self.subscribe()

    def signal_callback(self, string_msg):
        data = string_msg.data
        #! format is: object_name:banaana
        data = data.split(':')
        
        if len(data) == 1:
            command = data[0]
        elif len(data) == 2 and self.__object_name is None:
            self.__object_name = data[1]

            save_dir = os.path.join('.ros', self.__object_name + str(int(time.time())))
            self.__save_dir = os.path.join(os.environ['HOME'], save_dir)
            os.mkdir(self.__save_dir)
            rospy.loginfo('OPERATING DIRECTORY:\n{}'.format(self.__save_dir))

            #! make subdirectories
            self.__im_subdir = os.path.join(self.__save_dir, 'images')
            self.__mk_subdir = os.path.join(self.__save_dir, 'masks')
            os.mkdir(self.__im_subdir)
            os.mkdir(self.__mk_subdir)
            return
        elif len(data) == 2 and self.__object_name is not None:
            rospy.loginfo('OBJECT: {} IS SET'.format(self.__object_name))
            return
        
        if self.__object_name is None:
            rospy.logwarn('WHAT IS THE OBJECT NAME')
            return
        
        if command.lower() in ['annotate', 'model', 'save']:
            if self.__im_rgb is None or self.__im_mask is None:
                rospy.logwarn('IMAGES NOT SET')
                
            fn = str(int(time.time()))
            cv.imwrite(os.path.join(self.__im_subdir, fn + 'jpg'), self.__im_rgb)

            mk_dir = os.path.join(self.__mk_subdir, fn)
            os.mkdir(mk_dir)
            cv.imwrite(os.path.join(mk_dir, str(1) + 'png'), self.__im_mask)

            rospy.loginfo('SLEEPING FOR 5 SECONDS...')
            rospy.sleep(5)
            rospy.loginfo('AWAKE ...')
            
        elif command.lower() in  ['learn', 'train']:
            self.__model_path = self.train_zsd(self.__save_dir, self.__object_name)

            out_file = open(self.__text_fn, 'w')
            out_file.write(model_path + ' ' + self.__object_name)
            out_file.close()

        elif command.lower() in ['load']:
            self.load_trained_model(self.__model_path)
            
        elif command.lower() in  ['detect', 'run']:
            pass
            
    def image_callback(self, image_msg, mask_msg):
        self.__im_rgb = self.convert_image(img_msg, 'bgr8')
        self.__im_mask = self.convert_image(mask_msg, 'mono8')

        """
        _ , contours, _ = cv.findContours(im_mask, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
        if len(contours) is 0:
            return
        sorted(contours, key=lambda x: -len(x))
        """
        return
        

    def convert_image(self, image_msg, encoding = 'bgr8'):
        cv_img = None
        try:
            return self.__bridge.imgmsg_to_cv2(image_msg, encoding)
        except Exception as e:
            rospy.logerr(e)
            return
                                                        

    def subscribe(self):
        rospy.Subscriber('/qrcode_reader/output/string', String, self.signal_callback)

        image_sub = message_filters.Subscriber('image', Image)
        mask_sub = message_filters.Subscriber('mask', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, mask_sub], 5, 3)
        ts.registerCallback(self.image_callback)                
        

def main(argv):
    try:
        rospy.init_node('unknown_object_learning_action')
        uola = UnknownObjectLearningAction()
        rospy.spin()
    except rospy.ROSInternalException as e:
        rospy.logfatal(e)

if __name__ == '__main__':
    main(sys.argv)
