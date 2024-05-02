import numpy as np

def compute_entropy(p):
    entropy = 0.
    if p != 0 or p != 1:
        return - p * np.log2(p) - (1-p) * np.log2(1-p)
    return entropy


# split the node for each of the features
def feature_split(dataset, feature_index):
    left_node_indices = []
    right_node_indices = []

    for i,x in enumerate(X_train):
        if x[feature_index] == 1 :
            left_node_indices.append(i)
        else:
            right_node_indices.append(i)
    return left_node_indices, right_node_indices


# compute the weighted entropy in the splitted nodes
def weighted_entropy(X,y,left_data_indices, right_data_indices):
    left_weight = len(left_data_indices) / len(X)
    right_weight = len(right_data_indices) / len(X)
    left_proportion = sum(y[left_data_indices]) / len(left_data_indices) 
    right_proportion = sum(y[right_data_indices]) / len(right_data_indices) 
    return (left_weight * compute_entropy(left_proportion) + right_weight * compute_entropy(right_proportion))

# information gain = Entropy of Root Node - Weighted Entropy of child node
def information_gain(X, y, left_data_indices, right_data_indices):
    root_proportion = sum(y) / len(y)
    root_entropy = compute_entropy(root_proportion)
    weighted_entropy_child_node = weighted_entropy(X, y, left_data_indices, right_data_indices)
    return root_entropy - weighted_entropy_child_node


# Use One Hot Encoding to represent discrete values as 0 or 1 for false and true

# Calculate Decision Tree
X_train = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],[1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,0]])
y_train = np.array([1,1,0,0,1,1,0,1,0,0])


for i, feature_name in enumerate(['First Feature', 'Second Feature', 'Third Feature']):
    left_indices, right_indices = feature_split(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
