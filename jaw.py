#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jarvis ZHANG
@date: 2017/8/8
@editor: VS Code
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)


def show_example():
    img = plt.imread(os.path.join(sys.path[0], "testimage.jpg"))
    plt.imshow(img)
    plt.show()

def demo(fun, fun_inv, name):
    p = np.array([
        [ 81, 154],
 [ 78, 177],
 [ 79, 202],
 [ 81, 226],
 [ 88, 249],
 [ 98, 270],
 [115, 286],
 [135, 298],
 [157, 303],
 [180, 301],
 [201, 291],
 [220, 276],
 [235, 258],
 [245, 236],
 [251, 213],
 [255, 190],
 [257, 166],
 [ 94, 131],
 [107, 121],
 [125, 120],
 [143, 125],
 [157, 134],
 [186, 135],
 [202, 129],
 [219, 128],
 [236, 132],
 [247, 144],
 [170, 154],
 [168, 172],
 [167, 189],
 [165, 206],
 [145, 213],
 [154, 217],
 [164, 222],
 [174, 219],
 [184, 216],
 [111, 154],
 [122, 149],
 [135, 151],
 [147, 160],
 [133, 160],
 [121, 159],
 [192, 163],
 [204, 156],
 [216, 158],
 [225, 164],
 [216, 167],
 [204, 166],
 [124, 235],
 [141, 234],
 [154, 234],
 [164, 237],
 [175, 236],
 [187, 238],
 [201, 241],
 [186, 250],
 [173, 254],
 [161, 254],
 [151, 252],
 [139, 246],
 [130, 236],
 [153, 241],
 [163, 243],
 [174, 242],
 [195, 242],
 [173, 243],
 [162, 243],
 [153, 241]   ])
    q = np.array([
        [ 81, 154],
 [ 78, 177],
 [ 79, 202],
 [ 81, 226],
 [ 88, 249],
 [ 98, 270],
 [115, 286],
 [135, 298],
 [157, 303],
 [180, 301],
 [201, 291],
 [220, 276],
 [235, 258],
 [245, 236],
 [251, 213],
 [255, 190],
 [257, 166],
 [ 94, 131],
 [107, 121],
 [125, 120],
 [143, 125],
 [157, 134],
 [186, 135],
 [202, 129],
 [219, 128],
 [236, 132],
 [247, 144],
 [170, 154],
 [168, 172],
 [167, 189],
 [165, 206],
 [145, 213],
 [154, 217],
 [164, 222],
 [174, 219],
 [184, 216],
 [111, 154],
 [122, 149],
 [135, 151],
 [147, 160],
 [133, 160],
 [121, 159],
 [192, 163],
 [204, 156],
 [216, 158],
 [225, 164],
 [216, 167],
 [204, 166],
 [124, 235],
 [141, 234],
 [154, 234],
 [164, 237],
 [175, 236],
 [187, 238],
 [201, 241],
 [186, 250],
 [173, 254],
 [161, 254],
 [151, 252],
 [139, 246],
 [130, 236],
 [153, 241],
 [163, 243],
 [174, 242],
 [195, 242],
 [173, 243],
 [162, 243],
 [153, 241]
    ])
    image = plt.imread(os.path.join(sys.path[0], "testimage.jpg"))

    plt.figure(figsize=(8, 6))
    plt.subplot(231)
    plt.axis('off')
    plt.imshow(image)
    plt.title("Original Image")
    if fun is not None:
        transformed_image = fun(image, p, q, alpha=1, density=1)
        plt.subplot(232)
        plt.axis('off')
        plt.imshow(transformed_image)
        plt.title("%s Deformation \n Sampling density 1"%name)
        transformed_image = fun(image, p, q, alpha=1, density=0.7)
        plt.subplot(235)
        plt.axis('off')
        plt.imshow(transformed_image)
        plt.title("%s Deformation \n Sampling density 0.7"%name)
    if fun_inv is not None:
        transformed_image = fun_inv(image, p, q, alpha=1, density=1)
        plt.subplot(233)
        plt.axis('off')
        plt.imshow(transformed_image)
        plt.title("Inverse %s Deformation \n Sampling density 1"%name)
        transformed_image = fun_inv(image, p, q, alpha=1, density=0.7)
        plt.subplot(236)
        plt.axis('off')
        plt.imshow(transformed_image)
        plt.title("Inverse %s Deformation \n Sampling density  0.7"%name)

    plt.tight_layout(w_pad=0.1)
    plt.show()

def demo2(fun):
    ''' 
        Smiled Monalisa  
    '''
    
    p = np.array([
        [ 81, 154],
 [ 78, 177],
 [ 79, 202],
 [ 81, 226],
 [ 88, 249],
 [ 98, 270],
 [115, 286],
 [135, 298],
 [157, 303],
 [180, 301],
 [201, 291],
 [220, 276],
 [235, 258],
 [245, 236],
 [251, 213],
 [255, 190],
 [257, 166],
 [ 94, 131],
 [107, 121],
 [125, 120],
 [143, 125],
 [157, 134],
 [186, 135],
 [202, 129],
 [219, 128],
 [236, 132],
 [247, 144],
 [170, 154],
 [168, 172],
 [167, 189],
 [165, 206],
 [145, 213],
 [154, 217],
 [164, 222],
 [174, 219],
 [184, 216],
 [111, 154],
 [122, 149],
 [135, 151],
 [147, 160],
 [133, 160],
 [121, 159],
 [192, 163],
 [204, 156],
 [216, 158],
 [225, 164],
 [216, 167],
 [204, 166],
 [124, 235],
 [141, 234],
 [154, 234],
 [164, 237],
 [175, 236],
 [187, 238],
 [201, 241],
 [186, 250],
 [173, 254],
 [161, 254],
 [151, 252],
 [139, 246],
 [130, 236],
 [153, 241],
 [163, 243],
 [174, 242],
 [195, 242],
 [173, 243],
 [162, 243],
 [153, 241]
    ])
    q = np.array([
        [ 81, 154],
 [ 78, 177],
 [ 79, 202],
 [ 75, 200],
 [ 88, 249],
 [ 98, 270],
 [115, 286],
 [135, 298],
 [157, 303],
 [180, 301],
 [200, 290],
 [210, 270],
 [230, 250],
 [230, 230],
 [251, 213],
 [255, 190],
 [257, 166],
 [ 94, 131],
 [107, 121],
 [125, 120],
 [143, 125],
 [157, 134],
 [186, 135],
 [202, 129],
 [219, 128],
 [236, 132],
 [247, 144],
 [170, 154],
 [168, 172],
 [167, 189],
 [165, 206],
 [145, 213],
 [154, 217],
 [164, 222],
 [174, 219],
 [184, 216],
 [111, 154],
 [122, 149],
 [135, 151],
 [147, 160],
 [133, 160],
 [121, 159],
 [192, 163],
 [204, 156],
 [216, 158],
 [225, 164],
 [216, 167],
 [204, 166],
 [124, 235],
 [141, 234],
 [154, 234],
 [164, 237],
 [175, 236],
 [187, 238],
 [201, 241],
 [186, 250],
 [173, 254],
 [161, 254],
 [151, 252],
 [139, 246],
 [130, 236],
 [153, 241],
 [163, 243],
 [174, 242],
 [195, 242],
 [173, 243],
 [162, 243],
 [153, 241]
    ])
    image = plt.imread(os.path.join(sys.path[0], "testimage.jpg"))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image)
    transformed_image = fun(image, p, q, alpha=1, density=1)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(transformed_image)
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


if __name__ == "__main__":
    #affine deformation
    #demo(mls_affine_deformation, mls_affine_deformation_inv, "Affine")
    #demo2(mls_affine_deformation_inv)

    #similarity deformation
    #demo(mls_similarity_deformation, mls_similarity_deformation_inv, "Similarity")
    #demo2(mls_similarity_deformation_inv)

    #rigid deformation
    #demo(mls_rigid_deformation, mls_rigid_deformation_inv, "Rigid")
    demo2(mls_rigid_deformation_inv)
