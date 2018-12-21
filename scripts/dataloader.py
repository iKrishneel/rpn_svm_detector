#!/usr/bin/env python

import os
import sys
import math
import random
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm

"""
Class for computing intersection over union(IOU)
"""
class JaccardCoeff:

    def iou(self, a, b):
        i = self.__intersection(a, b)
        if i == 0:
            return 0
        aub = self.__area(self.__union(a, b))
        anb = self.__area(i)
        area_ratio = self.__area(a)/self.__area(b)        
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

class Dataloader(object):
    def __init__(self, data_dir = None, class_labels = 'class.txt', \
                 labels = 'masks', background_folder = None):

        assert os.path.isdir(data_dir), 'Invalid dataset directory! {}'.format(data_dir)

        class_labels = os.path.join(data_dir, class_labels)
        assert os.path.isfile(class_labels), 'Class label textfile not found!{}'.format(class_labels)

        labels_dir = os.path.join(data_dir, labels)
        assert os.path.isdir(labels_dir), 'Dataset labels folder not found: {}'.format(labels_dir)

        image_dir = os.path.join(data_dir, 'images')
        assert os.path.isdir(image_dir), 'Dataset image folder not found: {}'.format(image_dir)
        
        #! read the object labels
        lines = self.read_textfile(class_labels)
        class_label_dict = {}
        for line in lines:
            name, label = line.split(' ')
            class_label_dict[int(label)] = name

        self.image_ids = []

        #! read all labels folder
        self.__dataset = []
        for index, dfile in enumerate(os.listdir(labels_dir)):
            fn, ext = os.path.splitext(dfile)
            if len(ext) is 0:
                im_path = os.path.join(image_dir, fn + '.jpg')
                la_path = os.path.join(labels_dir, fn)
                if os.path.isfile(im_path):
                    #! read all label images in the folder
                    label_lists = []
                    class_ids = []
                    for llist in os.listdir(la_path):
                        #! filename is class id
                        class_id, _ = os.path.splitext(llist)
                        label_lists.append(os.path.join(labels_dir, os.path.join(fn, llist)))
                        class_ids.append(int(class_id))
                        
                    self.__dataset.append({'image': im_path, 'labels': label_lists, 'class_id': class_ids})
                    self.image_ids.append(index)
                else:
                    print ('Image not found! {}'.format(im_path))

        #! generate multiple instance
        self.__iou_thresh = 0.05
        self.__max_counter = 100
        self.__num_class = len(class_label_dict)
        self.__net_size = (800, 800)

        self.num_classes = len(class_label_dict) + 1
        self.source_class_ids = {'': [0], 'handheld_objects': np.arange(0, len(lines)+1)}

        self.DEBUG = False
        if self.DEBUG:
            self.__fetch_counter = 0
            for i in range(len(self.__dataset)):
                r = self.load_image(i, 1)
                print (r[1])
                x,y,w,h = r[1]
                cv.rectangle(r[0], (x, y), (w+x, h+y), (0, 255, 0), 3)
                cv.imshow('image', r[0])
                cv.waitKey(0)

    def get_data_size(self):
        return len(self.__dataset)
        
    def load_image(self, image_id, class_id=None):
        return self.argument(image_id, class_id)

    def argument(self, index, class_id=None):

        im_path = self.__dataset[index]['image']
        label_paths = self.__dataset[index]['labels']
        class_ids = self.__dataset[index]['class_id']

        #! read the rgb image
        im_rgb = cv.imread(im_path, cv.IMREAD_COLOR)

        masks = np.zeros((im_rgb.shape[0], im_rgb.shape[1], self.__num_class), np.bool)
        rects = np.zeros((1, 4, self.__num_class), np.int0)
        ordered_class_ids = np.zeros((self.__num_class), np.int0)

        for cls_id, lpath in zip(class_ids, label_paths):
            mk = cv.imread(lpath, cv.IMREAD_ANYDEPTH)
            _, contour = self.edge_contour_points(mk.copy())

            if contour is None:
                continue
            
            rect = cv.boundingRect(contour[0])
            
            rects[0, :, cls_id-1] = np.array(rect)
            
            mk = mk.astype(np.bool)
            masks[:, :, cls_id-1] = mk            
            ordered_class_ids[cls_id-1] = cls_id

        if class_id is not None:
            return im_rgb, rects[0, :, class_id-1]
            
        # overall bounding rect
        minx = miny = 1E9
        maxx = maxy = 0
        # for rect in rects:
        for i, cls_id in enumerate(ordered_class_ids):
            if cls_id == 0:
                continue
            print ([i, cls_id])
            x,y,w,h = rects[0, :, i]
            y2 = y+h
            x2 = x+w
            minx = x if x < minx else minx
            miny = y if y < miny else miny
            maxx = x2 if x2 > maxx else maxx
            maxy = y2 if y2 > maxy else maxy

        padx = random.randint(20, im_rgb.shape[1]-(maxx-minx))
        pady = random.randint(20, im_rgb.shape[0]-(maxy-miny))
        
        minx = minx-padx if minx-padx > 0 else 0
        miny = miny-pady if miny-pady > 0 else 0
        maxx = maxx+padx if maxx+padx < im_rgb.shape[1] else im_rgb.shape[1]
        maxy = maxy+pady if maxy+pady < im_rgb.shape[0] else im_rgb.shape[0]

        #! crop all to new size
        im_rgb = im_rgb[miny:maxy, minx:maxx]
        masks = masks[miny:maxy, minx:maxx]

        return im_rgb, masks, ordered_class_ids, rects, None



    
    def crop_image_dimension(self, image, rect, width, height):
        x = int((rect[0] + rect[2]/2) - width/2)
        y = int((rect[1] + rect[3]/2) - height/2)
        w = width
        h = height

        ## center 
        cx, cy = (rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0)
        shift_x, shift_y = (random.randint(0, int(w/2)), random.randint(0, int(h/2)))
        cx = (cx + shift_x) if random.randint(0, 1) else (cx - shift_x)
        cy = (cy + shift_y) if random.randint(0, 1) else (cy - shift_y)
        
        nx = int(cx - (w / 2))
        ny = int(cy - (h / 2))
        nw = int(w)
        nh = int(h)
        
        # img2 = image.copy()
        # cv.rectangle(img2, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 4)

        if nx > x:
            nx = x
            nw -=  np.abs(nx - x)
        if ny > y:
            ny = y
            nh -=  np.abs(ny - y)
        if nx + nw < x + w:
            nx += ((x+w) - (nx+nw))
        if ny + nh < y + h:
            ny += ((y+h) - (ny+nh))

        x = nx; y = ny; w = nw; h = nh
        # cv.rectangle(img2, (int(nx), int(ny)), (int(nx + nw), int(ny + nqh)), (255, 0, 255), 4)
        # cv.imshow("img2", img2)

        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        w = ((w - (w + x) - image.shape[1])) if x > image.shape[1] else w
        h = ((h - (h + y) - image.shape[0])) if y > image.shape[0] else h

        roi = image[int(y):int(y+h), int(x):int(x+w)].copy()
        new_rect = [int(rect[0] - x), int(rect[1] - y), rect[2], rect[3]]

        return roi, new_rect

    def resize_image_and_labels(self, image, rects, resize):
        img_list = []
        resize_rects = []
        for rect in rects:
            img = cv.resize(image, resize, cv.INTER_CUBIC)
            img_list.append(img)
            # resize label
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
                
            rect_resize = (int(xt), int(yt), int(xb - xt), int(yb - yt))
            resize_rects.append(rect_resize)
        return img, resize_rects


    def edge_contour_points(self, im_mask):
        if im_mask is None:
            return im_mask, None

        #! smooth the edges in mask
        # im_mask = cv.GaussianBlur(im_mask, (21, 21), 11.0)
        # im_mask[im_mask > 150] = 255

        im_mask2 = im_mask.copy()
        if len(im_mask2.shape) == 3:
            im_mask2 = cv.cvtColor(im_mask, cv.COLOR_BGR2GRAY)

        _, im_mask2 = cv.threshold(im_mask2, 127, 255, 0)        
        _ , contours, _ = cv.findContours(im_mask2, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
        
        #! remove noisy contours
        min_area = 16**2
        max_area = 0
        contour_obj = None
        for contour in contours:
            area = cv.contourArea(contour)
            if area < min_area:
                continue
            max_area, contour_obj = (area, contour) if area > max_area \
                                    else (max_area, contour_obj)

        # im_mask2 = cv.cvtColor(im_mask2, cv.COLOR_GRAY2BGR)
        # cv.drawContours(im_mask2, [contour_obj], 0, (0, 0, 255), 2)
        # x,y,w,h=cv.boundingRect(contour_obj)
        # cv.rectangle(im_mask2, (x,y), (x+w, y+h), (0, 255, 0), 3)
        # cv.imshow('mk', im_mask2)
            
        if max_area < min_area:
            return im_mask, None
        else:
            return im_mask, [contour_obj]

            
    @classmethod
    def read_textfile(self, filename):
        lines = [line.rstrip('\n')
                 for line in open(filename)                                                                             
        ]
        return np.array(lines)

    @classmethod
    def read_images(self, **kwargs):
        im_rgb = cv.imread(kwargs['image'], cv.IMREAD_COLOR)
        im_dep = cv.imread(kwargs['depth'], cv.IMREAD_ANYCOLOR)
        im_mask = cv.imread(kwargs['mask'], cv.IMREAD_GRAYSCALE)
        return [im_rgb, im_dep, im_mask]
    
    
        
    """
    Function flip image and rect around given axis
    """     
    def flip_image(self, image, rects, flip_flag = -1):
        im_flip = cv.flip(image, flip_flag)
        flip_rects = []
        for rect in rects:
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
            flip_rects.append(flip_rect)
        return im_flip, flip_rects

def main(argv):
    if len(argv) < 2:
        raise ValueError('Provide image list.txt')
    smi = Dataloader(argv[1])
    
if __name__ == '__main__':
    main(sys.argv)
