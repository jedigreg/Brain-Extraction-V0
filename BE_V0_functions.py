"""
Developed for the Image Analysis in Medicine Lab (IAMLAB)

Developer: Justin DiGregorio

Title: Brain Extraction, V0, Functions

"""

#%% import dependencies
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation
import scipy.ndimage.morphology as im
import copy
from skimage.measure import label
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy.spatial import distance

#%% function for performing post-processing on thresholded brain masks

def morph_brainMask(thresh_brain, selem1, selem2):
    
    # variable to store false positive reduced segmentations
    brain = copy.deepcopy(thresh_brain)
    
    # number of slices in thresholded volume
    num_slices = brain.shape[2]
    
    for j in range(num_slices):
        
        # erosion
        brain[:,:,j] = binary_erosion(brain[:,:,j], selem1)
    
    # connectivity analysis
    label_im = label(brain)
    num_regions = len(np.unique(label_im))
    region_sizes = []
    
    if num_regions > 0:
        for k in range(num_regions):
            labels = label_im.flatten()
            label_idx = np.where(labels == k)[0]
            region_sizes.append(len(label_idx))
                
    region_sizes[0] = 0
    brain_idx, brain_size = max(enumerate(region_sizes), key=(lambda x: x[1]))
    label_im[label_im != brain_idx] = 0
    label_im[label_im == brain_idx] = 1
    brain = label_im
        
    for j in range(num_slices):
        
        # dilation
        brain[:,:,j] = binary_dilation(brain[:,:,j], selem2)
        
        # hole filling
        brain[:,:,j] = im.binary_fill_holes(brain[:,:,j])
        
    return(brain)
    
#%% function for evaluating performance of V0 brain masks

def validate_brainMask(groundTruth, brainMask):
    
    A = groundTruth
    B = brainMask
    
    # convert ground truth and automated brain mask into 1D arrays
    A = A.reshape(A.shape[0]*A.shape[1], A.shape[2])
    A = A.flatten()
    
    B = B.reshape(B.shape[0]*B.shape[1], B.shape[2])
    B = B.flatten()
    
    # dice similiarity coefficient
    dsc = 1 - (distance.dice(A,B))
        
    # jaccard similiarity coefficient
    jsc = 1 - (distance.jaccard(A,B))
    
    # precision, recall, f-score
    prf = precision_recall_fscore_support(A,B)
    
    # accuracy
    acc = metrics.accuracy_score(A,B)
    
    # confusion matrix
    cm = confusion_matrix(A,B)
    
    return(dsc, jsc, prf, acc, cm)