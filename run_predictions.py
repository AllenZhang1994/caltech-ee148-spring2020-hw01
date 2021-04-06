import os
import numpy as np
import json
from PIL import Image


def detect_red_light(I, my_kernels):
    print(12345)
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    
    alpha = 0.99
    n_rows, n_cols, n_channels = np.shape(I)
    
    for kernel in my_kernels:
        box_height, box_width, _ = kernel.shape
        thredshold = int(np.tensordot(kernel, kernel, axes=([0,1,2], [0,1,2])))               
        for i in range(n_rows - box_height):
            for j in range(n_cols - box_width):
                window = I[i:i+box_height, j:j+box_width, :]
                score = int(np.tensordot(kernel, window, axes=([0,1,2], [0,1,2])))
                
                if score > alpha*thredshold:
                    bounding_boxes.append([i,j,i+box_height,j+box_width])
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes


# the following is the main code for handling data and do prediction
"""
Now save the convolution kernels to the kernels list
"""

# set the path to the convolution kernels
kernel_path = '/Users/yongzhezhang/Documents/CS148 Selected Topics in Computational Vision/Project-1/kernels'

os.makedirs(kernel_path,exist_ok=True) # create directory if needed 

# get sorted list of Kernels files: 
kernel_names = sorted(os.listdir(kernel_path)) 
kernel_Images = [f for f in kernel_names if '.jpg' in f]

my_kernels = []
for i in range(len(kernel_Images)):
    
    # read image using PIL:
    I = Image.open(os.path.join(kernel_path, kernel_Images[i]))
    I = np.asarray(I)[:,:,0:3]
    my_kernels.append(I)

"""
Finished Saving the convolution kernels to the kernels list
"""

# set the path to the downloaded data: 
data_path = '/Users/yongzhezhang/Documents/CS148 Selected Topics in Computational Vision/Project-1/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '/Users/yongzhezhang/Documents/CS148 Selected Topics in Computational Vision/Project-1/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of Image files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 


# main progrom for prediction
preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I, my_kernels)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
