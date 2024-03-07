import numpy as np

def compute_entropy(y):
    entropy = 0.
    if len(y) != 0:
        p1 = len(y[y==1]) / len(y)
        if p1 !=0 and p1 != 1:
            entropy = -1 * p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
    return entropy


def feature_split(dataset, node_indices, feature_num):
    left_node_indices = []
    right_node_indices = []

    for i in node_indices:
        if dataset[i][feature_num] == 1 :
            left_node_indices.append(i)
        else:
            right_node_indices.append(i)
    return left_node_indices, right_node_indices

def information_gain(X, y, node_indices, feature):
    left_indices, right_indices = feature_split(X, node_indices, feature)
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    h_node = compute_entropy(y_node)
    h_left = compute_entropy(y_left)
    h_right = compute_entropy(y_right)

    w_left = len(y_left) / len(y_node)
    w_right = len(y_right) / len(y_node)

    return h_node - ((h_left * w_left) + (h_right * w_right))

def best_split_feature(X, y, node_indices):
    best_feature = -1
    max_gain = 0
    feature_count = X.shape[1]
    for i in range(feature_count):
        info_gain = information_gain(X, y, node_indices, i)  
        if info_gain > max_gain :
            max_gain = info_gain
            best_feature = i

    return best_feature

tree = []
def build_tree_recursive (X, y ,  node_indices, branch_name, max_depth, current_depth):
    if current_depth == max_depth:
        print(f"{branch_name} leaf node with indices : {node_indices}")
        return
    best_feature = best_split_feature(X, y, node_indices)
    print(f"Depth {current_depth}, {branch_name}: Split on Feature: {best_feature}")

    left_indices, right_indices = feature_split(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)



# Calculate Decision Tree
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

build_tree_recursive(X_train, y_train, range(len(y_train)), "Root", 2, 0)