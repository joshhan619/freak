import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import math


"""
Implementation and performance of FREAK in the application of Panoramic Image Construction. FREAK algorithms are based
on the research paper "FREAK: Fast Retina Keypoint" by Alahi, Ortiz, and Vandergheynst.

FREAK: Fast Retina Keypoint

A. Alahi, R. Ortiz, and P. Vandergheynst. FREAK: Fast Retina Keypoint. In IEEE Conference on Computer Vision and Pattern
Recognition, 2012. Alexandre Alahi, Raphael Ortiz, Kirell Benzi, Pierre Vandergheynst
Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland
"""

def __descriptor_help(image, ksize, pts, p):
    """
    Helper function in select_pairs. Blur an image for a specific set of points using a Gaussian kernel with a set
    patch size and find the intensities at these points.

    :param image: the image that the keypoint belongs to
    :param ksize: patch width such that a patch will be 2*ksize + 1 by 2*ksize + 1 in size
    :param pts: array of receptive field centres with the same patch size
    :param p: the keypoint where the centre of the sampling pattern lies
    :return: array of smoothed intensities for this subset of receptive field centres
    """
    out = []
    for pt in pts:
        if pt[0] + p[0] - ksize < 0 or pt[0] + p[0] + ksize >= image.shape[1] or \
                pt[1] + p[1] - ksize < 0 or pt[1] + p[1] + ksize >= image.shape[0]:
            return -1
        src = image[pt[1] + p[1] - ksize:pt[1] + p[1] + ksize + 1, pt[0] + p[0] - ksize:pt[0] + p[0] + ksize + 1]
        gauss = cv2.GaussianBlur(src, (2*ksize + 1, 2*ksize + 1), 2, borderType=cv2.BORDER_CONSTANT)
        out.append(gauss[ksize, ksize])

    return out


def compute_orientation(pts, intensities):
    """
    Compute the orientation of a feature from the FREAK sampling pattern intensities.

    :param pts: array of receptive field centres
    :param intensities: the smoothed intensities of each receptive field centre for a single feature
    :return: the orientation of the feature in radians
    """
    M = 42
    pairs = [(0, 2), (1, 3), (2, 4), (3, 5), (0, 4), (1, 5)]
    sum = np.zeros((2, ))
    for i in range(7):
        rank_pts = pts[6 * i: 6 * i + 6]
        rank_i = intensities[6 * i: 6 * i + 6]
        for j in range(6):
            first_pt = np.array(rank_pts[pairs[j][0]])
            second_pt = np.array(rank_pts[pairs[j][1]])
            vec = first_pt - second_pt
            norm = np.linalg.norm(vec)
            vec = vec / norm
            sum += (rank_i[pairs[j][0]] - rank_i[pairs[j][1]]) * vec

    gradients = sum / M
    return math.atan(gradients[1] / gradients[0])


def __calculate_intensities(image, pts, p):
    """
    Helper function in select_pairs. Calculates the smoothed intensities of a list of receptive field centres for a
    single keypoint.

    :param image: the image that the keypoint belongs in
    :param pts: array of receptive field centres with coordinates relative to the centre of the sampling pattern
    :param p: the keypoint where the centre of the sampling pattern lies
    :return: array of intensities for each of the 43 receptive field centres. Returns -1 if we cannot calculate the
    Gaussian at a receptive field centre.
    """
    intensities = np.array([], dtype='int64')
    radii = [18, 13, 9, 6, 4, 3, 2, 1]
    smoothed_intensity = -1

    for i in range(8):
        if i == 7:
            smoothed_intensity = __descriptor_help(image, radii[i], [pts[42]], p)
        else:
            smoothed_intensity = __descriptor_help(image, radii[i], pts[6 * i: 6 * i + 6], p)

        if smoothed_intensity == -1:
            return -1

        intensities = np.concatenate((intensities, smoothed_intensity))

    return intensities


def select_pairs(images, keypoints_arr, corr_thresh):
    """
    Selects the best sampling pairs from the FREAK sampling pattern based on the set of image keypoints and a
    correlation threshold.

    :param images: array of images
    :param keypoints_arr: array of keypoints, with each keypoint corresponding to the image at the same index in images
    :param corr_thresh: correlation threshold when selecting best sampling pairs
    :return: array of large descriptors, best sampling pairs selection mask, reduced array of keypoints
    """
    descriptors = []
    sums = np.array([0 for i in range(903)], dtype='int64')
    new_kps = [[] for i in range(10)]

    # Receptive field centres
    pts = [(33, 0), (17, -30), (-17, -30), (-33, 0), (-17, 30), (17, 30),
           (22, 13), (22, -13), (0, -26), (-22, -13), (-22, 13), (0, 26),
           (18, 0), (9, -17), (-9, -17), (-18, 0), (-9, 17), (9, 17),
           (11, 7), (11, -7), (0, -13), (-11, -7), (-11, 7), (0, 13),
           (8, 0), (4, -8), (-4, -8), (-8, 0), (-4, 8), (4, 8),
           (5, 3), (5, -3), (0, -6), (-5, -3), (-5, 3), (0, 6),
           (4, 0), (2, -4), (-2, -4), (-4, 0), (-2, 4), (2, 4),
           (0, 0)]

    for x in range(len(images)):
        image = images[x]
        keypoints = keypoints_arr[x]
        for kp in keypoints:
            p = [int(kp.pt[0]), int(kp.pt[1])]

            # Calculate intensities from the sampling pattern
            intensities = __calculate_intensities(image, pts, p)

            if isinstance(intensities, int) and intensities == -1:
                continue

            # Calculate orientation and rotate sampling pattern
            orientation = compute_orientation(pts, intensities)
            cosine = math.cos(orientation)
            sine = math.sin(orientation)
            rotation_matrix = np.array([[cosine, -sine], [sine, cosine]])
            new_pts = []
            for pt in pts:
                new_pt = np.matmul(rotation_matrix, np.array(pt))
                new_pts.append((int(new_pt[0]), int(new_pt[1])))

            # Recalculate intensities
            intensities = __calculate_intensities(image, new_pts, p)
            if isinstance(intensities, int) and intensities == -1:
                continue

            # Compute large descriptors and find column sums
            t = 0
            des = []
            for i in range(43):
                for j in range(43):
                    if i > j:
                        if intensities[i] - intensities[j] > 0:
                            sums[t] += 1
                            des.append(1)
                        else:
                            des.append(0)
                        t += 1

            new_kps[x].append(kp)

            descriptors.append(des)

    # Form initial mask by ordering columns by high variance (mean = 0.5)
    num_kps = sum([len(keypoints) for keypoints in new_kps])
    sums = np.divide(sums, num_kps)
    sums = np.add(sums, -0.5)
    sums = np.abs(sums)
    mask = np.argsort(sums)

    # Correlation check
    corr_mask = [0 for i in range(903)]
    corr_mask[0] = 1
    descriptors = np.array(descriptors)

    good_pairs = [0]
    for i in range(1, 903):
        for j in range(len(good_pairs)):
            corrcoeff = np.corrcoef(descriptors[:, mask[i]], descriptors[:, mask[good_pairs[j]]])
            if abs(corrcoeff[0][1]) >= corr_thresh:
                break

            if j == len(good_pairs) - 1:
                corr_mask[i] = 1
                good_pairs.append(i)

    out_mask = []
    for i in range(903):
        if corr_mask[i] == 1:
            out_mask.append(mask[i])

    for i in range(903):
        if corr_mask[i] == 0:
            out_mask.append(mask[i])

    return descriptors, np.array(out_mask), new_kps


def compute_descriptor(descriptors, mask):
    """
    Calculate the FREAK descriptor given a set of large descriptors and a selection mask from the training step.

    :param descriptors: array of large descriptors (length 903) from the training step
    :param mask: a selection mask for the large descriptors with the best sampling pairs at the front of the
    descriptor
    :return: array of descriptors (length 512) with the best sampling pairs at the front of the descriptor
    """
    for i in range(len(descriptors)):
        descriptors[i] = descriptors[i][mask]

    descriptors = np.packbits(descriptors, axis=-1)

    return descriptors[:512]


def stitch_images(images, descriptor_func):
    """
    Construct a panorama from a set of images using a given descriptor function. The set of images must be ordered and
    the panorama will be a horizontal panorama from the first image as the leftmost image and the last image as the
    rightmost image.

    :param images: array of images to stitch together;
    :param descriptor_func: function that will find the matches between features using a specific type of descriptor.
    :return: the stitched panorama image
    """
    orb_obj = cv2.ORB_create()
    kps = [None, None]
    image1 = images[5]
    for i in range(6, 10):
        image2 = images[i]

        kps[0] = orb_obj.detect(image2, None)
        kps[1] = orb_obj.detect(image1, None)
        image_group = [cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY), cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)]

        matches, kps, src_pts, dest_pts, time = descriptor_func(image_group, kps, False)

        H, Hmask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC, 5.0)

        # Calculate where the new right-most corners will be after the transformation
        corner1 = np.matmul(H, np.array([image2.shape[1], 0, 1]))
        corner2 = np.matmul(H, np.array([image2.shape[1], image2.shape[0], 1]))
        min_x = min(corner1[0] / corner1[2], corner2[0] / corner2[2])
        min_x = max(min_x, image1.shape[1])

        image_result = cv2.warpPerspective(image2, H, (int(min_x), image1.shape[0]))
        # plt.subplot(121)
        # plt.imshow(image1, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(image_result, cmap='gray')
        # plt.show()
        image_result[0:image1.shape[0], 0:image1.shape[1]] = image1
        # plt.imshow(image_result, cmap='gray')
        # plt.show()

        image1 = image_result

    # Left side stitch
    for i in range(5):
        image2 = images[4 - i]
        kps[0] = orb_obj.detect(image2, None)
        kps[1] = orb_obj.detect(image1, None)
        image_group = [cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY), cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)]

        matches, kps, src_pts, dest_pts, time = descriptor_func(image_group, kps, False)

        H, Hmask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC, 5.0)

        # Calculate the new left-most corners after the transformation
        corner1 = np.matmul(H, np.array([0, 0, 1]))
        corner2 = np.matmul(H, np.array([0, image2.shape[0], 1]))
        min_x = -max(corner1[0] / corner1[2], corner2[0] / corner2[2])

        # Perform transformations
        if min_x > 0:
            T = np.float32([[1, 0, min_x], [0, 1, 0], [0, 0, 1]])
            # Translate image to the right to avoid being cropped
            image2 = cv2.warpPerspective(image2, T, (image1.shape[1] + int(min_x), image1.shape[0]))

        image_result = cv2.warpPerspective(image2, H, (image1.shape[1] + int(min_x), image1.shape[0]))
        # plt.subplot(121)
        # plt.imshow(image1)
        # plt.subplot(122)
        # plt.imshow(image_result, cmap='gray')
        # plt.show()
        image_result[0:image1.shape[0], int(min_x):image1.shape[1] + int(min_x)] = image1
        # plt.imshow(image_result, cmap='gray')
        # plt.show()

        image1 = image_result

    return image1


def FREAK_compute_and_match(images, kps, timed):
    """
    Find the FREAK descriptors for each keypoint in a pair of images and determine the matches between these images.

    :param images: an array of two images to be matched
    :param kps: the keypoints of the two images
    :param timed: a boolean that will print the timing results of the FREAK algorithms if equal to 1
    :return: a set of matches found between the two images, a reduced list of keypoints, an array of keypoints matched
    in the first image, an array of keypoints matched in the second image, and the description time (equal to 0 if
    timed = 0)
    """
    des_time = 0

    start = time.perf_counter()
    large_des, mask, kps = select_pairs(images, kps, 0.2)
    end = time.perf_counter()
    if timed:
        print(f"FREAK select pairs training time: {end - start:0.4f} seconds")

    # freak_obj = cv2.xfeatures2d.FREAK_create()
    start = time.perf_counter()
    descriptors1 = compute_descriptor(large_des[:len(kps[0]), :], mask)
    descriptors2 = compute_descriptor(large_des[len(kps[0]):len(kps[0]) + len(kps[1]), :], mask)
    # kps[0], descriptors1 = freak_obj.compute(images[0], kps[0])
    # kps[1], descriptors2 = freak_obj.compute(images[1], kps[1])
    end = time.perf_counter()
    if timed:
        print(f"FREAK description time: {end - start:0.4f} seconds\n")
        des_time = end - start

    descriptors1 = np.uint8(descriptors1)
    descriptors2 = np.uint8(descriptors2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kps[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dest_pts = np.float32([kps[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return matches, kps, src_pts, dest_pts, des_time


def SIFT_compute_and_match(images, kps, timed):
    """
    Find the SIFT descriptors for each keypoint in a pair of images and determine the matches between these images.

    :param images: an array of two images to be matched
    :param kps: the keypoints of the two images
    :param timed: a boolean that will print the timing results of the SIFT algorithm if equal to 1
    :return: a set of matches found between the two images, a reduced list of keypoints, an array of keypoints matched
    in the first image, an array of keypoints matched in the second image, and the description time (equal to 0 if
    timed = 0)
    """
    des_time = 0

    start = time.perf_counter()
    sift_obj = cv2.xfeatures2d.SIFT_create()
    kps[0], descriptors1 = sift_obj.compute(images[0], kps[0])
    kps[1], descriptors2 = sift_obj.compute(images[1], kps[1])
    end = time.perf_counter()
    if timed:
        print(f"SIFT description time: {end - start:0.4f} seconds\n")
        des_time = end - start

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    good = sorted(good, key=lambda x: x[0].distance)
    src_pts = np.float32([kps[0][m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dest_pts = np.float32([kps[1][m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return good, kps, src_pts, dest_pts, des_time


def BRISK_compute_and_match(images, kps, timed):
    """
    Find the SIFT descriptors for each keypoint in a pair of images and determine the matches between these images.
    :param images: an array of two images to be matched
    :param kps: the keypoints of the two images
    :param timed: a boolean that will print the timing results of the BRISK algorithm if equal to 1
    :return: a set of matches found between the two images, a reduced list of keypoints, an array of keypoints matched
    in the first image, an array of keypoints matched in the second image, and the description time (equal to 0 if
    timed = 0)
    """
    des_time = 0

    start = time.perf_counter()
    brisk_obj = cv2.BRISK_create()
    kps[0], descriptors1 = brisk_obj.compute(images[0], kps[0])
    kps[1], descriptors2 = brisk_obj.compute(images[1], kps[1])
    end = time.perf_counter()
    if timed:
        print(f"BRISK description time: {end - start:0.4f} seconds\n")
        des_time = end - start

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kps[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dest_pts = np.float32([kps[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return matches, kps, src_pts, dest_pts, des_time


def main():
    # Load images
    images = []
    for i in range(10):
        image = cv2.imread("./images/image_{0}.png".format(i + 1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    ##########################################
    # Rotation Test
    print("ROTATION TEST:")
    times = []
    orb_obj = cv2.ORB_create()
    image_rotate = cv2.resize(images[0], (int(images[0].shape[1]*0.6), int(images[0].shape[0]*0.6)))
    image_rotate = cv2.rotate(image_rotate, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    kps = []
    kps.append(orb_obj.detect(images[0]))
    kps.append(orb_obj.detect(image_rotate))

    image_rotate_group = [cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY), cv2.cvtColor(image_rotate, cv2.COLOR_RGB2GRAY)]

    ## FREAK ##
    matches, kps, src_pts, dest_pts, freak_time = FREAK_compute_and_match(image_rotate_group, kps, True)
    times.append(freak_time)

    matches_drawing = cv2.drawMatches(images[0], kps[0], image_rotate, kps[1], matches[:15], None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                                      matchColor=(0, 255, 255))
    plt.title("FREAK matches between image_1.png and a rotated version of image_1.png")
    plt.imshow(matches_drawing)
    plt.show()

    ## SIFT ##
    matches, kps, src_pts, dest_pts, sift_time = SIFT_compute_and_match(image_rotate_group, kps, True)
    times.append(sift_time)

    matches_drawing = cv2.drawMatchesKnn(images[0], kps[0], image_rotate, kps[1], matches[:15], None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                                         matchColor=(0, 255, 255))
    plt.title("SIFT matches between image_1.png and a rotated version of image_1.png")
    plt.imshow(matches_drawing)
    plt.show()

    ## BRISK ##
    matches, kps, src_pts, dest_pts, brisk_time = BRISK_compute_and_match(image_rotate_group, kps, True)
    times.append(brisk_time)

    matches_drawing = cv2.drawMatches(images[0], kps[0], image_rotate, kps[1], matches[:15], None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                                         matchColor=(0, 255, 255))
    plt.title("BRISK matches between image_1.png and a rotated version of image_1.png")
    plt.imshow(matches_drawing)
    plt.show()

    # Show times
    plt.bar(["FREAK", "SIFT", "BRISK"], times)
    plt.xlabel("Feature description algorithms")
    plt.ylabel("Description time (seconds)")
    plt.show()
    #########################################
    # Panorama Test
    print("PANORAMA TEST:")
    times = []

    ## FREAK ##
    start = time.perf_counter()
    panorama = stitch_images(images, FREAK_compute_and_match)
    end = time.perf_counter()
    print(f"FREAK panoramic image construction time: {end - start:0.4f} seconds\n")
    times.append(end - start)

    plt.imshow(panorama)
    plt.title("FREAK Panorama Image Construction")
    plt.show()

    ## SIFT ##
    start = time.perf_counter()
    panorama = stitch_images(images, SIFT_compute_and_match)
    end = time.perf_counter()
    print(f"SIFT panoramic image construction time: {end - start:0.4f} seconds\n")
    times.append(end - start)

    plt.imshow(panorama)
    plt.title("SIFT Panorama Image Construction")
    plt.show()

    ## BRISK ##
    start = time.perf_counter()
    panorama = stitch_images(images, BRISK_compute_and_match)
    end = time.perf_counter()
    print(f"BRISK panoramic image construction time: {end - start:0.4f} seconds\n")
    times.append(end - start)

    plt.imshow(panorama)
    plt.title("BRISK Panorama Image Construction")
    plt.show()

    # Show times
    plt.bar(["FREAK", "SIFT", "BRISK"], times)
    plt.xlabel("Feature description algorithms")
    plt.ylabel("Panorama image construction time (seconds)")
    plt.show()


if __name__ == '__main__':
    main()