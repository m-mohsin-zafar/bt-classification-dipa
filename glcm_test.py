"""
This Python File is intended to get an understanding of Grey Level Co-occurrence Matrix and
Texture Descriptors/Features that may be calculated from GLCM
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from PIL import Image


'''
Angles -    refer to pixel pairs based on angular orientation. 
            Typical are 0, 45, 90, and 135 degrees.
            Also, the paper we intend to implement mention all four of these angles.

Python Documentation describes this input argument as;
    List of pixel pair angles in radians
'''
ANGLES = [0., np.pi / 4., np.pi / 2., 3. * np.pi / 4.]
'''
Distances - refers to as Distance offset to be considered in determining Pair Relationship.
            A single offset describes 1 pixel to the right and 1 pixel to the left.
            I think that the offset can be multi-dimensional but haven't tried out yet.
            We can give multiple distances in form of List as input argument to 'greycomatrix(...)' function
            *Also, the paper we intend to follow doesn't describe what and how many offsets to use.
            
Python Documentation describes this input argument as;
    List of pixel pair distance offsets
'''
# DISTANCES = [1, 2]
DISTANCES = [1]
'''
Properties - refers to as Texture Descriptors Calculated from Co-occurrence Matrix.
            6 Texture Descriptors can be calculated via Inbuilt Function as of now.
            *Also, the paper we intend to follow uses 4 out of 6 Descriptors which are given below.
'''
properties = ["correlation", "contrast", "homogeneity", "energy"]

'''
mpimg.imread -  The Img variable simply stores 2D Image array, this is similar to MATLAB's imread([Path\][Filename])
                This function is available in 'matplotlib' module of python
'''
Img = mpimg.imread(r'F:\University Data\PIEAS\First Semester\DIPA\Brain Tumor Dataset\L1\1.BMP')
# plt.imshow(Img)
Img = np.array(Img)

'''
Image.open -    Referred function reads image and also has many rich attributes, the output is not just a simple 2D Image array
                This function is available in 'PILLOW' module of python, PILLOW Module has very extensive image processing functions
'''
# I = Image.open(r'F:\University Data\PIEAS\First Semester\DIPA\Brain Tumor Dataset\L1\1.BMP')
# I = np.array(I)

'''
Returns:
    The grey-level co-occurrence histogram. 
    The value `P[i,j,d,theta]` is the number of times that grey-level `j` occurs at a distance `d` and at an angle `theta` from grey-level `i`. 
    If `normed` is `False`, the output is of type uint32, otherwise it is float64. 
    The dimensions are: levels x levels x number of distances x number of angles.
    
Example:
    Say, in our case; 
        Levels = 256
        Angles = 4
        Distances = 1
    So, the output dimensions will be [256 x 256 x 1 x 4]
    
Arguments: "symmetric" and "normed" are false by default. if we put symmetric = true then it accumulates values irrespective 
of the order of pixels. If order is to be preserved then this must be false
'''
glcm = greycomatrix(Img, distances=DISTANCES, angles=ANGLES, levels=256, symmetric=True, normed=True)
'''
greycoprops(...):

Purpose:
    Calculate texture properties of a GLCM.
    Compute a feature of a grey level co-occurrence matrix to serve as a compact summary of the matrix.
    
Returns:
    2-dimensional array. `results[d, a]` is the property 'prop' for the d'th distance and the a'th angle

Example:
    Say in our case;
        Distances = 1
        Angles = 4
    So, the output Dimensions [1 x 4]

'''
prop = greycoprops(glcm, properties[1])
# stats = []
# for k in range(n_img):
#     glcm = greycomatrix(Img[k], distances=DISTANCES, angles=ANGLES, levels=256, symmetric=True, normed=True)

# prop = [np.mean(greycoprops(glcm, properties[i])) for i in range(len(properties))]

# stats.append(prop)
# stats = np.array(stats)
result = np.array(glcm).flatten()
non_zero_result = result[result != 0]
non_zero_result = np.transpose(non_zero_result)
print(non_zero_result)
