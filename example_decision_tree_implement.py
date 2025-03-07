# decision tree implementation


# %%
import numpy as np

X_train = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]
)
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])

print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:", type(X_train))

print("First few elements of y_train:", y_train[:5])
print("Type of y_train:", type(y_train))

print("The shape of X_train is:", X_train.shape)
print("The shape of y_train is: ", y_train.shape)
print("Number of training examples (m):", len(X_train))

# %%
# UNQ_C1
# GRADED FUNCTION: compute_entropy


def compute_entropy(y):
    """
    Computes the entropy for

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """
    # You need to return the following variables correctly
    entropy = 0.0

    ### START CODE HERE ###

    if len(y) == 0:
        return 0

    count = 0
    for i in range(len(y)):
        if y[i] == 1:
            count += 1

    p_1 = count / len(y)

    if p_1 == 0 or p_1 == 1:
        return 0
    else:
        entropy = -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)

    ### END CODE HERE ###

    return entropy


# UNQ_C2
# GRADED FUNCTION: split_dataset


def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """

    # You need to return the following variables correctly
    left_indices = []
    right_indices = []

    ### START CODE HERE ###
    Y = X[node_indices]

    for i, x in enumerate(Y):
        if x[feature] == 1:
            left_indices.append(node_indices[i])
        else:
            right_indices.append(node_indices[i])
    return left_indices, right_indices

    ### END CODE HERE ###

    return left_indices, right_indices


# %%
# UNQ_C3
# GRADED FUNCTION: compute_information_gain


def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        cost (float):        Cost computed

    """
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    # You need to return the following variables correctly
    information_gain = 0

    ### START CODE HERE ###

    # p_node = sum(y_node)/len(y_node)
    h_node = compute_entropy(y_node)

    w_left = len(left_indices) / len(X_node)
    w_right = len(right_indices) / len(X_node)
    # p_left = sum(y_node[left_indices])/len(left_indices)
    # p_right = sum(y_node[right_indices])/len(right_indices)

    weighted_entropy = w_left * compute_entropy(y_left) + w_right * compute_entropy(
        y_right
    )

    information_gain = h_node - weighted_entropy

    ### END CODE HERE ###

    return information_gain


# %%
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

# %%
# UNQ_C4
# GRADED FUNCTION: get_best_split


def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """

    # Some useful variables
    num_features = X.shape[1]

    # You need to return the following variables correctly
    best_feature = -1

    ### START CODE HERE ###
    information_gain = 0

    for i in range(num_features):
        information_gain_tmp = compute_information_gain(X, y, node_indices, i)
        if information_gain_tmp > information_gain:
            information_gain = information_gain_tmp
            best_feature = i

    ### END CODE HERE ##

    return best_feature


# %%

best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)


# %%
tree = []


def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree.
        current_depth (int):    Current depth. Parameter used during recursive call.

    """

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " " * current_depth + "-" * current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices)

    formatting = "-" * current_depth
    print(
        "%s Depth %d, %s: Split on feature: %d"
        % (formatting, current_depth, branch_name, best_feature)
    )

    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth + 1)


# %%
build_tree_recursive(
    X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0
)
# generate_tree_viz(root_indices, y_train, tree)
