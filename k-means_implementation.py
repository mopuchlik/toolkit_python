
    #
    # In pseudocode, the K-means algorithm is as follows:
    #
    #   # Initialize centroids
    #   # K is the number of clusters
    #   centroids = kMeans_init_centroids(X, K)
    #
    #   for iter in range(iterations):
    #       # Cluster assignment step:
    #       # Assign each data point to the closest centroid.
    #       # idx[i] corresponds to the index of the centroid
    #       # assigned to example i
    #       idx = find_closest_centroids(X, centroids)
    #
    #       # Move centroid step:
    #       # Compute means based on centroid assignments
    #       centroids = compute_centroids(X, idx, K)
    #


# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###

    for i in range(X.shape[0]):
        x = X[i][0]
        y = X[i][1]

        cost = np.inf

        for j in range(K):

            cx = centroids[j][0]
            cy = centroids[j][1]

            cost_tmp = (x - cx)**2 + (y - cy)**2

            if cost_tmp < cost:
                cost = cost_tmp

                idx[i] = j

     ### END CODE HERE ###

    return idx

# UNQ_C2
# GRADED FUNCTION: compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    ### START CODE HERE ###

    for i in range(K):

        X_i = X[idx == i]

        k = X_i.shape[0]

        x_sum = 0
        y_sum = 0

        for j in range(k):

            x_sum += X_i[j][0]
            y_sum += X_i[j][1]

        x_avg = x_sum / k
        y_avg = y_sum / k

        centroids[i][0] = x_avg
        centroids[i][1] = y_avg

    ### END CODE HERE ##

    return centroids

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):

        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show()
    return centroids, idx


def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids


