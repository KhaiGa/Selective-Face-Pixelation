# Selective Face Pixelation

### Introduction
Pixelate individual faces in videos by using one of two face association methods: **Input Association**
or **Cluster Association**.  
Face Detection and Recognition utilize the InsightFace python-library.
##### Input Association:
Compares input images of target faces to all face tracks and pixelates those with a feature similarity above 
a similarity threshold. 
##### Cluster Association:
Clusters all tracks according to their average face embedding similarities using 
Hierarchical Agglomerative Clustering.
### Installation
EDIT: Insightface 0.1.5 auto-download link for the model does not work anymore. 
To fix, download the ResNet50 model from https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ, extract and place it under .insightface/models.

Install the CUDA version of mxnet for GPU usage and faster performance.  
For other packages, see requirements.txt.
### Usage
Create the following folder/s:  
    - Selective-Face-Pixelation/Video (place video to be pixelated here)    
    - Selective-Face-Pixelation/input_imgs (only for Input Association)     
Then simply run main.py and follow the console instructions.
##### Input Association:
Sample images of faces to be pixelated should be placed into /input_imgs. Make certain images only contain
one face at a time and depict facial features clearly.      
(Tip: Providing images of the same person featuring varying head poses often improves accuracy.)

Lowering the similarity threshold will correctly pixelate target faces more often but may also 
increase the rate of pixelations of non-target faces. Increasing the similarity threshold has the opposite 
effect.

##### Cluster Association:
Coming soon...
