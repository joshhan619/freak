# 2020 Summer CSC420 Final Project
## Viability of FREAK in Panoramic Image Construction
*Matthew Dans, Joshua Han, Asad Jamil*

### Summary
The purpose of this project is to implement the FREAK algorithm in order to test its performance for panoramic image construction applications. 
The implementation of FREAK is based on the paper: FREAK: Fast Retina Keypoint[1] by Alahi et al. 

Two tests are performed:
1. Rotation-Invariance Test: A rotated and rescaled version of an image is produced. Features in the original image and its rotated variant are matched using 
three different keypoint descriptor methods: FREAK, SIFT, and BRISK. The results of each method's matches are shown and their description times are compared.

2. Panoramic Image Construction: Panoramas are constructed using FREAK, SIFT and BRISK keypoint descriptors. Their timings are recorded and compared.

1. A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern
Recognition, 2012. Alexandre Alahi, Raphael Ortiz, Kirell Benzi, Pierre Vandergheynst
Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland

### How to Run

1. Make sure that the latest versions of opencv-python, NumPy, and Matplotlib are installed. 
2. In order to use the OpenCV implementation of SIFT, a non-free algorithm, an older version of opencv-contrib-python needs to be installed. 
This can be done in the following ways, with pip and Anaconda respectively: 
`
pip install opencv-contrib-python==3.4.2.16
`

`
conda install opencv-contrib-python=3.4.2.16
`

3. Open a terminal and navigate to the location freak.py was extracted.
4. Run `python freak.py`
