import numpy as np
from collections import Counter

class NodeTree:
	'''
	Data structure for tree
	'''
	def __init__(self, feat_split=None, threshold=None, left=None, right=None, *, label=None, depth=None):
		# Initialize the Node class
		self.feat_split = feat_split # store feature index used for splitting the tree
		self.threshold = threshold # threshold for each feature split
		self.left = left # left part of the tree
		self.right = right # right part of the tree
		self.label = label # store label value
		self.depth = depth # current depth of the tree
	
	def is_leaf_node(self):
		# If label is None, then it is not a leaf node yet
		return(self.label is not None)


class DecisionTree:
	'''
	Class for Decision Tree binary classifier
	'''
	def __init__(self, min_samples_split=2, max_depth=10):
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.root = None

	def fit(self, X, y):
		'''
		Grow the decision tree recursively starting from the root
		'''
		self.root = self._build_tree(X, y)

	def predict(self, X):
		'''
		Traverse the tree recursively to create prediction
		'''
		pred_res = []
		for x in X:
			pred_res.append(self._traverse_tree(x, self.root))
		return(np.array(pred_res))

	def _build_tree(self, X, y, depth=0):
		'''
		Recursively build decision tree until it reach terminal state
		Terminal states: 
		1. If depth already greater than max depth, 
		2. If there is already 1 label, 
		3. If number of sample is already smaller then minimum sample split
		'''
		n_samples, n_features = X.shape
		n_labels = len(np.unique(y))

		if ((depth >= self.max_depth) or (n_labels == 1) or (n_samples < self.min_samples_split)):
			# Terminal state for the recursive function will return most common label stored as label for the leaf node
			return(NodeTree(label=self._most_common_label(y)))
		else: 
			# recursive function
			
			# select the best split according to gini impurity, use _find_best_split function
			best_feat, best_thresh = self._find_best_split(X, y)

			# recursively create a new Node object with best feature split and and its threshold
			left_part_split = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
			right_part_split = np.argwhere(X[:, best_feat] > best_thresh).flatten()
			left = self._build_tree(X[left_part_split, :], y[left_part_split], depth+1)
			right = self._build_tree(X[right_part_split, :], y[right_part_split], depth+1)

			return(NodeTree(best_feat, best_thresh, left, right))

	def _most_common_label(self, y):
		counter = Counter(y)
		most_common = counter.most_common(1)[0][0]
		return(most_common)

	def _find_best_split(self, X, y):
		'''
		Function to find the best feature and threshold to be used to split the tree
		'''
		split_col = None
		split_threshold = None

		n_samples, n_features = X.shape
		n_labels = len(np.unique(y))
		
		# Set default value of curr_gini to the gini impurity value of the parent node
		parent_unique_labels = np.unique(y, return_counts=True)[1]
		curr_gini = 1.0
		for n in parent_unique_labels:
			curr_gini += -np.sum((n / n_samples) ** 2)

		# Temporary variable for y that will be used for concatenating
		temp_y = y.reshape(y.shape[0],1)

		# check for each feature value, which value at which feature index gives the best information gain
		for col in range(n_features):
			# Temporary variable for X per each feature that will be used for concatenating
			temp_X = X[:,col].reshape(n_samples,1) 
			all_data = np.concatenate((temp_X,temp_y), axis=1)

			# Sort the data using the column with the feature to get sorted feature values for each class
			sorted_data = all_data[np.argsort(all_data[:,0])]

			 # Split the data back into threshold and classes values
			thresholds, obs_classes = np.array_split(sorted_data, 2, axis = 1)
			obs_classes = obs_classes.astype(int)
			
			# Temporary variable to track how the split is being done in each node
			num_left = np.zeros((n_labels), dtype=np.float64)
			num_right = parent_unique_labels.copy()

			for i in range(1, n_samples):
				class_ = obs_classes[i - 1][0]
				num_left[class_] += 1
				num_right[class_] -= 1
				
				# Get gini impurity value for each splitted node
				gini_left = 1.0
				gini_right = 1.0
				for x in range(n_labels):
					gini_left += -np.sum((num_left[x] / i) ** 2)
					gini_right += -np.sum((num_right[x] / (n_samples - i)) ** 2)

				# get weighted average impurity of two child nodes
				gini = (i * gini_left + (n_samples - i) * gini_right) / n_samples

				# If 2 equal values result in a split, avoid doing the split
				if thresholds[i][0] == thresholds[i - 1][0]:
					continue

				# get the feature index and feature value when gini value is lower than current gini, indicating highest information gain
				if gini < curr_gini:
					curr_gini = gini # update the curr_gini with the best observed gini currently
					split_col = col
					split_threshold = (thresholds[i][0] + thresholds[i - 1][0]) / 2
		
		# Return the index of best column to be used for split and the threshold for splitting
		return(split_col, split_threshold)

	def _traverse_tree(self, x, node):
		'''
		Function to traverse the decision tree that has been built
		Check if input x is given, which route it should take in the tree
		Recursively traverse the tree until it finds a leaf node
		'''
		if(node.is_leaf_node()):
			return(node.label) # Terminal state for the recursive function
		elif (x[node.feat_split] <= node.threshold):
			return(self._traverse_tree(x, node.left))
		else: return(self._traverse_tree(x, node.right))
