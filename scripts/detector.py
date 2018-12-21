#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2 as cv
import random
import rospy
import matplotlib.pylab as plt
import pickle

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mask_rcnn')
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import AvgPool2D, Lambda, Input, Flatten, MaxPooling2D

from sklearn.neighbors import NearestNeighbors
from sklearn import svm

from dataloader import Dataloader

class JaccardCoeff:
    """
    Class for computing intersection over union(IOU)
    """
    def iou(self, a, b):
        i = self.__intersection(a, b)
        if i == 0:
            return 0
        aub = self.__area(self.__union(a, b))
        anb = self.__area(i)
        score = anb/aub
        return score
        
    def __intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0:
            return 0
        else:
            return (x, y, w, h)
        
    def __union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def __area(self, rect):
        return np.float32(rect[2] * rect[3])

    

class InferenceConfig(Config):

    # Give the configuration a recognizable name
    NAME = "handheld_objects"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    NUM_CLASSES = 81

    
class RPNSVMDetector(object):

    def __init__(self, model_path, is_train=True, log_dir=None):

        assert os.path.isfile(model_path), 'INVALID MODEL FILE: {}'.format(model_path)

        if log_dir is not None:
            assert os.path.isdir(log_dir), 'INVALID LOGGING DIRECTOR. CHECK THAT IT EXIST!'
        
        log_dir = os.path.join(os.environ['HOME'], '.ros/logs') if log_dir is None else log_dir
        
        #! get the detector configuration
        config = InferenceConfig()
        config.display()

        print('Initializing the Mask RCNN model')
        self.__mrcnn_model = modellib.MaskRCNN(mode='inference', config=config, model_dir=log_dir)
        self.__mrcnn_model.load_weights(model_path, by_name=True)
        print('Model is successfully initialized!')

        self.__prob_thresh = 0.8
        self.__jc = JaccardCoeff()
        self.__input_shape = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)
        self.__model = self.build_model()

        self.__trained_zsd = None
        save_fname = 'test_model.p'

        if is_train:
            dataset_dir = '/home/krishneel/Documents/datasets/2018-09-19-14-26-11'
            self.train(dataset_dir, save_fname)
            self.load_trained_model(save_fname)

            # dataset_dir = '/home/krishneel/Documents/datasets/wrs/2018-09-18-20-14-23'
            # dataloader = Dataloader(dataset_dir)            
            
            for i in range(dataloader.get_data_size()):
                input_image, rect = dataloader.load_image(i, class_id=1)
                self.detect(input_image)
            
    def build_model(self):
        layer_name=['ROI', 'roi_align_classifier']
        _model = self.__mrcnn_model.keras_model
        model = Model(inputs=_model.input, outputs=[_model.get_layer(layer_name[0]).output, \
                                                    _model.get_layer(layer_name[1]).output])
        in_shape = model.output_shape[1][2:]
        in_feats = Input(shape=in_shape)
        # pool = MaxPooling2D(pool_size=(7, 7), strides=1, padding='valid', name='pool')(in_feats)
        pool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool')(in_feats)
        flat = Flatten(name='flat')(pool)
                
        # flat = self.sigmoid(name='sigmoid')(flat)
        self.__model_1 = Model(inputs=in_feats, outputs=flat)

        print (self.__model_1.summary())

        return model

    def sigmoid(self, **kwargs):
        def layer(x):
            return 1.0/(1.0 + tf.keras.backend.exp(-x))
        return Lambda(layer, **kwargs)

        
    def predict(self, image):
        molded_image, image_meta, window = self.__mrcnn_model.mold_inputs([image])
        anchors = self.__mrcnn_model.get_anchors(molded_image[0].shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
        rois_output, rpn_output = self.__model.predict([molded_image, image_meta, anchors], verbose=0)

        #! reduce the rpn features via pooling 
        roi_feats = self.__model_1.predict(rpn_output[0])
        
        # return rois_output[0], rpn_output[0]
        return rois_output[0], roi_feats

    def load_trained_model(self, save_fname):
        #! check that model file exist
        assert os.path.isfile(save_fname), 'ERROR TRAINED MODEL FILE NOT FOUND: {}'.format(save_fname)
        
        print('Reading model from file')
        self.__trained_zsd = pickle.load(open(save_fname, 'rb'))
        print ('Successful read the model from file')
        return True

    def detect(self, input_image, threshold=1e6):

        input_image = cv.resize(input_image, self.__input_shape, cv.INTER_CUBIC)
        rois_output, roi_feats = self.predict(input_image)
        
        print ('Predicting...')
        distances, neighbors = self.__trained_zsd.kneighbors(roi_feats, return_distance=True)
        
        min_index = np.argmin(distances[:, 0])
        if distances[min_index] > threshold:
            return None
                
        print ('Min Dist: {}'.format(distances[min_index]))
        y1,x1,y2,x2 = rois_output[min_index] * self.__input_shape[0]
        box = np.array([x1, y1, x2-x1, y2-y1], np.int0)
        cv.rectangle(input_image, (x1,y1), (x2,y2), (0, 255,0), 2)

        #! get top 5 results
        sorted_indices = np.argsort(distances[:, 0])[:5]
        for si in sorted_indices[1:]:
            y1,x1,y2,x2 = rois_output[si] * self.__input_shape[0]
            box = np.array([x1, y1, x2-x1, y2-y1], np.int0)
            cv.rectangle(input_image, (x1,y1), (x2,y2), (0, 0, 255), 1)

        # labels = self.__trained_zsd.predict(roi_feats)
        # indices,  = np.where(labels==1)
        # if indices.shape[0] == 0:
        #     return
        # for index in indices:
        #     print (index)
        #     y1,x1,y2,x2 = rois_output[index] * self.__input_shape[0]
        #     box = np.array([x1, y1, x2-x1, y2-y1], np.int0)
        #     cv.rectangle(input_image, (x1,y1), (x2,y2), (0, 255,0), 2)
        
        # cv.imshow('img', input_image)
        # cv.waitKey(3)

        return np.array([x1, y1, x2, y2], np.int0)
        
    def train(self, dataset_dir, save_fname, pyrd_level=2):
        #! load dataset
        # dataloader = DatasetLoader(dataset_dir)
        dataloader = Dataloader(dataset_dir)
        sample_size = dataloader.get_data_size()

        #! firstly extract features
        training_features = []
        non_object_features = []
        for i in range(sample_size):
            print ('Index: {}'.format(i))
            
            #! load the full size image, object rect and the label
            image, obj_rect = dataloader.load_image(i, class_id=1)
            if obj_rect[2] == 0 or obj_rect[3] == 0:
                continue
            #! todo: crop and resize to generate more samples
            images, rects = self.image_pyramid(image, obj_rect, 2, pyrd_level)
            
            for im, rect in zip(images, rects):
                #! resize the image and bbox to the network input size
                # input_image, bbox = self.resize_image_and_bbox(im, rect)
                #! run through detector
                # rois_output, rpn_output = self.predict(input_image)
                rois_output, roi_feats = self.predict(im)
                #! reduce the rpn features via pooling 
                # roi_feats = self.__model_1.predict(rpn_output)
                
                #! get postive rois covering positive
                for index, roi in enumerate(rois_output):
                    y1,x1,y2,x2 = roi*self.__input_shape[0]
                    box = np.array([x1, y1, x2-x1, y2-y1], np.int0)
                    
                    #! save rois whose IOU with GT is greater than threshold
                    iou = self.__jc.iou(rect, box)
                    if iou > self.__prob_thresh:
                        training_features.append(roi_feats[index])
                        # cv.rectangle(input_image, (x1,y1), (x2,y2), (0, 255,0), 2)
                    else:
                        non_object_features.append(roi_feats[index])

        print ('Training Classifier')
        #! build nearest neigbor        
        nearest_neigbors = NearestNeighbors(n_neighbors=7, algorithm='ball_tree')
        nearest_neigbors.fit(training_features)
        print('Saving model to disk')
        save_fname = os.path.join(dataset_dir, save_fname)
        pickle.dump(nearest_neigbors, open(save_fname, 'wb'))

        # svm_model = svm.OneClassSVM(kernel="linear")
        # svm_model.fit(training_features)
        # print('Saving model to disk')
        # pickle.dump(svm_model, open(save_fname, 'wb'))

        return save_fname
                
            
    def image_pyramid(self, image, rect, scale=2, levels=2):
        images = [image]
        rects = [rect]
        s = scale
        factor = 2
        for level in range(levels):
            bbox = rect.copy().astype(np.float32)
            bbox[0:2] -= (bbox[2:4]/factor) * s
            bbox[2:4] *= s
            bbox = bbox.astype(np.int0)

            bbox[0] = 0 if bbox[0] < 0 else bbox[0]
            bbox[1] = 0 if bbox[1] < 0 else bbox[1]
            bbox[2] -= bbox[2]+bbox[0]-image.shape[1] if bbox[2]+bbox[0] > image.shape[1] else 0
            bbox[3] -= bbox[3]+bbox[1]-image.shape[0] if bbox[3]+bbox[1] > image.shape[0] else 0

            x,y,w,h = bbox
            im_roi = image[y:y+h, x:x+w].copy()
            s *= factor

            #! correct the rect of the object
            a,b,c,d = rect
            a = a-x
            b = b-y
            im_roi, bbox = self.resize_image_and_bbox(im_roi, np.array([a, b, c, d]))
            
            images.append(im_roi)
            rects.append(bbox)

            # x,y,w,h = bbox
            # cv.rectangle(im_roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
            # cv.imshow('roi', im_roi)
            
            for fflag in [-1, 0, 1]:
                im_flip, flip_rect = self.flip_image(im_roi, bbox, fflag)
                images.append(im_flip)
                rects.append(flip_rect)
                
                # x,y,w,h = flip_rect
                # cv.rectangle(im_flip, (x, y), (x+w, y+h), (0, 255, 0), 3)
                # cv.imshow('flip', im_flip)
                # cv.waitKey(0)
        
        return images, rects
    

    def resize_image_and_bbox(self, image, rect):
        """
        Function to resize image and the labels
        """
        print ('IN: {} {}'.format(self.__input_shape, image.shape))
        img = cv.resize(image, self.__input_shape, cv.INTER_CUBIC)
        
        ratio_x = np.float32(image.shape[1]) / np.float32(img.shape[1])
        ratio_y = np.float32(image.shape[0]) / np.float32(img.shape[0])

        x = np.float32(rect[0])
        y = np.float32(rect[1])
        w = np.float32(rect[2])
        h = np.float32(rect[3])
            
        xt = x / ratio_x
        yt = y / ratio_y
        xb = (x + w) / ratio_x
        yb = (y + h) / ratio_y
            
        # rect_resize = (int(xt), int(yt), int(xb - xt), int(yb - yt))
        rect_resize = np.array([xt, yt, xb-xt, yb-yt], np.int0)
        return img, rect_resize

    def flip_image(self, image, rect, flip_flag):
        im_flip = cv.flip(image, flip_flag)
        pt1 = (rect[0], rect[1])
        pt2 = (rect[0] + rect[2], rect[1] + rect[3])
        if flip_flag is -1:
            pt1 = (image.shape[1] - pt1[0] - 1, image.shape[0] - pt1[1] - 1)
            pt2 = (image.shape[1] - pt2[0] - 1, image.shape[0] - pt2[1] - 1)
        elif flip_flag is 0:
            pt1 = (pt1[0], image.shape[0] - pt1[1] - 1)
            pt2 = (pt2[0], image.shape[0] - pt2[1] - 1)
        elif flip_flag is 1:
            pt1 = (image.shape[1] - pt1[0] - 1, pt1[1])
            pt2 = (image.shape[1] - pt2[0] - 1, pt2[1])

        x = min(pt1[0], pt2[0])
        y = min(pt1[1], pt2[1])
        w = np.abs(pt2[0] - pt1[0])
        h = np.abs(pt2[1] - pt1[1])

        x = 0 if x < 0 else x
        y = 0 if y < 0 else y

        flip_rect = [x, y, w, h]
        return im_flip, flip_rect
    
    
    def visualize_filters(self, dataset, show=True):
        """Take an array of shape (n, height, width) or (n, height, width, 3)
        and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
        dataset = dataset.transpose((2, 0, 1))
        dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        n = int(np.ceil(np.sqrt(dataset.shape[0])))
        padding = (((0, n ** 2 - dataset.shape[0]), (0, 1), (0, 1)) + ((0, 0),) * (dataset.ndim - 3))
        dataset = np.pad(dataset, padding, mode='constant', constant_values=1)
        dataset = dataset.reshape((n, n) + dataset.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, dataset.ndim + 1)))
        dataset = dataset.reshape((n * dataset.shape[1], n * dataset.shape[3]) + dataset.shape[4:])
        if show:
            plt.imshow(dataset); plt.axis('off')
            plt.show()
        return dataset

        
def main(argv):
    zssd = RPNSVMDetector(argv[1])
    # dl = DatasetLoader(argv[1])

if __name__ == '__main__':
    main(sys.argv)
