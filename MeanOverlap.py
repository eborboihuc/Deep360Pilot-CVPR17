#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Hou-Ning Hu and Yen-Chen Lin

from collections import namedtuple
from math import pi


class MeanOverlap(object):
    """ Class for Mean Overlap metric via a simplified IoU (Only for relative comparison)
        
        Init:
            W, H        - Panoramic image size, int, float, in pixel format
            FOV         - FOV angle, float, in degree format
        
        Input: 
            center1     - Viewpoint one, (x, y), int, float, can be in degree, radius or pixel format
            center2     - Viewpoint two, (x, y), int, float, can be in degree, radius or pixel format
            is_centered - A boolean, if is_centered, shift the coordinate center to left top

        Functions:
            IOU_deg     - Calculate IOU in degree format
            IOU         - Calculate IOU in pixel format

        Output:
            IoU         - Simplified version of Mean Overlap (In order to speed up)
    """

    def __init__(self, W=1920, H=960, FOV=65.625):
        self.Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
        
        # Pano image size
        assert W > 0
        assert H > 0
        assert FOV > 0 and FOV < 180, "FOV ({}) should be in range of [0, 180]".format(FOV)
        self.W = W
        self.H = H

        # We use 4:3 as normal FOV format in VR player, and this give a 282.5 * 350 pixels in 1920 * 960 Pano.
        self.WIDTH = self.deg_to_pix(W, FOV)
        #self.HEIGHT = 0.75 * self.WIDTH
        
        # We use 1:1 as normal FOV format in VR headset, and this give a 350 * 350 pixels in 1920 * 960 Pano.
        self.HEIGHT = self.WIDTH

    def deg_to_pix(self, width, degree):
        return width*degree / 360.0

    def rad_to_pix(self, width, radius):
        return width*radius / (2*pi)

    def pix_to_deg(self, width, pixel):
        return 360.0 * pixel/width

    def pix_to_rad(self, width, pixel):
        return 2*pi * pixel/width

    def deg_to_rad(self, degree):
        return degree * pi / 180.0

    def rad_to_deg(self, radius):
        return radius * 180.0 / pi

    def area(self, a, b):  # returns 0 if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx >= 0) and (dy >= 0):
            return dx*dy
        else:
            return 0

    def IOU_deg(self, center1, center2, is_centered=False):
        x1, y1 = center1
        x2, y2 = center2
        
        # If is_centered, shift coordinate to start from left-top
        if is_centered:
            x1 = (x1 + 180.0) % 360.0
            x2 = (x2 + 180.0) % 360.0
            y1 += 90.0
            y2 += 90.0
        
        x1 = self.deg_to_pix(self.W, x1)
        y1 = self.deg_to_pix(self.W, y1)
        x2 = self.deg_to_pix(self.W, x2)
        y2 = self.deg_to_pix(self.W, y2)
        
        return self.IOU((x1, y1), (x2, y2))

    def IOU(self, center1, center2, is_centered=False):

        x1, y1 = center1
        x2, y2 = center2
        
        # If is_centered, shift coordinate to start from left-top
        if is_centered:
            x1 = (x1 + self.W/2.0) % self.W
            x2 = (x2 + self.W/2.0) % self.W
            y1 += self.H/2.0
            y2 += self.H/2.0

        r1 = []
        r2 = []

        # Simplified version of IoU for speed up
        # exceed boundary case
        if x1+self.WIDTH/2 >= self.W:
            r1.append(self.Rectangle(x1-self.WIDTH/2, y1-self.HEIGHT/2, self.W, y1+self.HEIGHT/2))
            r1.append(self.Rectangle(0, y1-self.HEIGHT/2, x1+self.WIDTH/2 - self.W, y1+self.HEIGHT/2))

        if x2+self.WIDTH/2 >= self.W:
            r2.append(self.Rectangle(x2-self.WIDTH/2, y2-self.HEIGHT/2, self.W, y2+self.HEIGHT/2))
            r2.append(self.Rectangle(0, y2-self.HEIGHT/2, x2+self.WIDTH/2 - self.W, y2+self.HEIGHT/2))

        if x1-self.WIDTH/2 < 0:
            r1.append(self.Rectangle(0, y1-self.HEIGHT/2, x1+self.WIDTH/2, y1+self.HEIGHT/2))
            r1.append(self.Rectangle(x1-self.WIDTH/2 + self.W, y1-self.HEIGHT/2, self.W, y1+self.HEIGHT/2))

        if x2-self.WIDTH/2 < 0:
            r2.append(self.Rectangle(0, y2-self.HEIGHT/2, x2+self.WIDTH/2, y2+self.HEIGHT/2))
            r2.append(self.Rectangle(x2-self.WIDTH/2 + self.W, y2-self.HEIGHT/2, self.W, y2+self.HEIGHT/2))

        # normal case
        if len(r1) == 0:
            r1.append(self.Rectangle(x1-self.WIDTH/2, y1-self.HEIGHT/2, x1+self.WIDTH/2, y1+self.HEIGHT/2))
        if len(r2) == 0:
            r2.append(self.Rectangle(x2-self.WIDTH/2, y2-self.HEIGHT/2, x2+self.WIDTH/2, y2+self.HEIGHT/2))

        acc = 0.0
        for x in r1:
            for y in r2:
                acc += self.area(x, y)

        return acc / (self.WIDTH*self.HEIGHT)

if __name__ == '__main__':
    mo = MeanOverlap(W=1920, H=960)
    
    assert abs(mo.IOU_deg((10, 20), (10, 65), is_centered=True) - mo.IOU((1013, 587), (1013, 827))) < 1e-10, \
            "IOU_deg is: {} while IOU is: {}".format(mo.IOU_deg((10, 20), (10, 65), is_centered=True), mo.IOU((1013, 587), (1013, 827)))
    assert abs(mo.IOU_deg((10, 20), (42.8125, 20)) - 0.5) < 1e-10, \
            "IOU is: {}".format(mo.IOU_deg((10, 20), (42.8125, 20)))
    assert abs(mo.IOU_deg((15, 20), (15, 20), is_centered=True) - 1.0) < 1e-10, \
            "IOU is: {}".format(mo.IOU_deg((15, 20), (15, 20)))
    assert mo.IOU((10, 20), (361, 20)) == 0.0, "IOU is: {}".format(mo.IOU((10, 20), (361, 20)))
    assert mo.IOU((410, 20), (410, 20)) == 1.0, "IOU is: {}".format(mo.IOU((410, 20), (410, 20)))
    assert mo.IOU((1745, 20), (165, 20)) == 3500.0/122500, "IOU is: {}".format(mo.IOU((1745, 20), (165, 20)))
    # Enable this if using 4:3
    #assert mo.IOU((165, 20), (165, 282.5)) == 0.0, "IOU is: {}".format(mo.IOU((165, 20), (165, 282.5)))
    
    print "Success!"
