import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float
from math import inf

# Clustering Methods


def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    for n in range(num_iters):
        # YOUR CODE HERE
        distances = np.zeros(k)
        for i in range(N):
            datapoint = features[i]
            count = 0
            for center in centers:
                distances[count] = np.linalg.norm(datapoint-center)
                count += 1
            assignments[i] = distances.argmin()
        bins = np.zeros([k, D])
        points = np.zeros(k)
        for i in range(N):
            cluster = int(assignments[i])
            bins[cluster] += features[i]
            points[cluster] += 1
        newcenters = np.zeros(centers.shape)
        for i in range(k):
            newcenters[i] = bins[i]/points[i]
        if np.allclose(newcenters, centers):
            break
        else:
            centers = newcenters
        # END YOUR CODE

    return assignments


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    for n in range(num_iters):
        # YOUR CODE HERE
        workingfeatures = np.tile(features, (k, 1))
        workingcenters = np.repeat(centers, N, axis=0)
        """Broadcasting features to an numpy array of N X k and 
        working centers to a numpy array of N X k so that the 
        computation of distance from each center can be done by simply 
        subtracting the elements in the array"""
        assignments = np.argmin(np.sum(
            ((workingcenters - workingfeatures)**2) ** (1/2), axis=1).reshape(k, N), axis=0)
        """Here we subtracted the corresponding entries of the centers and the feature points, squared them and added them 
        via the np.sum method, and then we took the square root of each sum to obtain the distance between the cluster
        center and the feature point. We reshaped the array to the dimensions k X N such that the i,j th entry of the
        matrix represents distance of the ith feature point to the jth cluster center. Then we take the index of the smallest 
        entry along each row, thus getting the assignments"""
        oldcenters = centers.copy()
        for j in range(k):
            centers[j] = np.mean(features[assignments == j], axis=0)
        if np.allclose(oldcenters, centers):
            break
        # END YOUR CODE
    return assignments


def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N
    while n_clusters > k:
        # YOUR CODE HERE
        distances = squareform(pdist(centers))
        distances = np.where(distances != 0, distances, inf)
        """Using the pdist and the squareform function to calculate the distance between the centers
        Replacing the zero distance computed as the distance between the same point with infinity using the np.where method
        as we want to merge to different clusters, not the same one with itself"""
        closest_center_index = np.argmin(
            distances)  # finding the closest cluster centers in the flattened distances matrix
        i = closest_center_index // n_clusters
        # finding the two-dimentional indices of the closest clusters in the distances array
        j = closest_center_index - i * n_clusters
        if i > j:
            i, j = j, i
        for y in range(N):
            if assignments[y] == j:
                # reassigning all the points that were the part of cluster j to cluster i
                assignments[y] = i
        # deleting the cluster center j from the array of cluster centers
        centers = np.delete(centers, j, axis=0)
        n_clusters -= 1
        for y in range(N):
            if assignments[y] > j:
                # since we just deleted a cluster from a set of clusters, we need to shift down the number representing
                assignments[y] -= 1
                # the cluster center of all points assigned to a cluster whose number is greater that j by 1
        cluster_members = []
        for y in range(N):
            if assignments[y] == i:
                cluster_members.append(features[y])
        centers[i] = np.mean(np.array(cluster_members), axis=0)
    return assignments

# Pixel-Level Features


def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    # YOUR CODE HERE
    features = img.reshape(H * W, C)
    # END YOUR CODE

    return features


def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))
    # YOUR CODE HERE
    locations = np.dstack(np.mgrid[0 : H, 0 : W]).reshape((H * W, 2))
    features[:, 0 : C] = color.reshape((H * W, C))
    features[:, C : C + 2] = locations
    features = (features - np.mean(features, axis = 0)) / np.std(features, axis = 0)#normaling the feature vectors so that the 
    #distances between the feature vectors will depend on real difference and won't be influenced by the mere presence of larger numbers
    # END YOUR CODE
    return features


def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    # YOUR CODE HERE
    pass
    # END YOUR CODE
    return features


# Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    # YOUR CODE HERE
    accuracy = np.mean(mask == mask_gt)
    # END YOUR CODE

    return accuracy


def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
