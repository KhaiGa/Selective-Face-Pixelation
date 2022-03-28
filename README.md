# Selective Face Pixelation
![final](https://user-images.githubusercontent.com/50239198/160453937-971c49b3-7f31-4ea1-80bc-d32906ad2756.png)
### Introduction
Pixelate individual faces (while retaining all others) in videos by using one of two face association methods: **Input Association**
or **Cluster Association**.  
Face Detection and Recognition utilize the InsightFace python-library.
##### Input Association:
Compares input images of target faces to all face tracks and pixelates those with a feature similarity above 
a similarity threshold. 
##### Cluster Association:
Clusters all tracks according to their average face embedding similarities using 
Hierarchical Agglomerative Clustering.
### Installation
EDIT: Insightface 0.1.5 auto-download link for detection and recognition models do not work anymore. 
To fix, download the RetinaFace-R50 model (Detection) and R100 mxnet model (Recognition) from [here](https://github.com/deepinsight/insightface/tree/master/model_zoo)
, extract and copy files to .insightface/models/retinaface_r50_v1 and .insightface/models/arcface_r100_v1, respectively. <br/>
Download and install ffmpeg from here: https://www.ffmpeg.org/download.html <br/>
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
