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

A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern
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

### Conclusion
Based on our findings, the FREAK algorithm as implemented is not sufficient to be used in place of other algorithms (SIFT, BRISK) for feature-based registration in a panoramic image construction pipeline. Although FREAK matched the other algorithms in terms of accuracy and was faster at matching keypoints, the pair selection process involved prior to generating FREAK descriptors made it significantly slower than the other algorithms. FREAK out performs other feature descriptors in the field of object detect due to its ability to pre-train the optimal pair selection for a specific object, which can then be used to quickly generate and compare feature descriptors for that object in any other image. However, this does not translate to PIC as the keypoints in each pair of overlapping images are typically unique to those images, meaning the pair selection algorithm needs to be run for every pair of images adding substantial runtime overhead. As the feature descriptor generating and comparison aspects of FREAK are significantly faster than other descriptors, we believe further investigation should be done to optimize the pair selection aspect of FREAK, specifically for PIC. 

More details about previous literature on this topic, our implementation, experiment setup, and results can be found in **Report.pdf** in this repository.
