"""
Developed for the Image Analysis in Medicine Lab (IAMLAB)

Developer: Justin DiGregorio

Title: Brain Extraction, V0, Main Script

"""

#%% import dependencies
import scipy.io as spio
import os
import copy
from skimage import img_as_ubyte
from skimage.morphology import disk
from BE_V0_functions import morph_brainMask, validate_brainMask

#%% preallocate memory and define global variables

# directory containing .mat FLAIR MRI volumes
volDir = ""

# directory containing .mat binary ground truth brain masks - **if available**
gtDir = ""
perform_validation = 'False'

# directory to save automated brain masks in
saveDir = ""

# data file names of number of volumes
file_names = os.listdir(volDir)
num_vols = len(file_names)

# structural elements for morphological processing
eroder = disk(4)
dilater = disk(6)

#%% perform brain extraction

for i in range(num_vols):  
    
    print('Performing brain extraction on volume', i+1)
    
    # load volume
    data = spio.loadmat(volDir + file_names[i], struct_as_record=False, squeeze_me=True)
    data = data["im"]
    v = data.final
    
    # create copy of volume to store brain segmentation
    brain = copy.deepcopy(v)
    
    print('Thresholding...')
    
    # apply thresholding
    brain[brain < 200] = 0
    brain[brain > 400] = 0
    
    # binarize thresholded volume
    brain[brain > 0] = 1
    
    print('Morphological processing...')
    
    # apply morphological processing
    brain = morph_brainMask(brain, eroder, dilater)
    
    # save results
    brainMask = img_as_ubyte(brain > 0)
    spio.savemat(os.path.join(saveDir, file_names[i]), {'brainMask':brainMask})

#%% evaluate performance - **run this section if corresponding ground truths are available**

if (perform_validation == 'True'):
    
    # lists for storage of validation metrics
    DSCs = []
    JSCs = []
    PRFs = []
    ACCs = []
    CMs = []

    for i in range(num_vols):
        
        # load ground truth
        gt = spio.loadmat(gtDir + file_names[i])
        gt = gt["gt"]
        
        # load automated brain mask
        brain = spio.loadmat(saveDir + file_names[i])
        brain = brain["brainMask"]
        
        # generate validation metrics
        dsc, jsc, prf, acc, cm = validate_brainMask(gt, brain)
        
        DSCs.append(dsc)
        JSCs.append(jsc)
        PRFs.append(prf)
        ACCs.append(acc)
        CMs.append(cm)
        
    # compute average performance based on validation metrics
    avgDSC = sum(DSCs)/len(DSCs)
    avgJSC = sum(JSCs)/len(JSCs)
    avgACC = sum(ACCs)/len(ACCs)
 