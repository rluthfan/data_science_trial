import numpy as np
from collections import Counter

def euclidean_distance(features, predict, order=None):
	'''
	Calculate vector norm
	'''
	return(np.linalg.norm(np.array(features)-np.array(predict), ord=order))

class KNN:
	'''
	Class for k-Nearest Neighbors classifier
	'''

	def __init__(self, k=3, l_norm_used=2):
		self.k = k
		self.l_norm_used = l_norm_used

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		'''
		Use _predict function to predict the whole dataset
		'''
		pred_res = []
		for x in X:
			pred_res.append(self._predict(x))

		return(np.array(pred_res))

	def _predict(self, x):
		'''
		Give prediction for each data point
		'''

		# Initialize distance
		distances = []
		# Compute distances between x and all examples in the training set
		for x_train in self.X_train:
			distances.append(euclidean_distance(x_train, x, self.l_norm_used))
		
		# Sort by distance and return indices of the first k neighbors
		k_idx = np.argsort(distances)[:self.k]

		# Extract the labels of the k nearest neighbor training samples
		# Initialize neighbor labels in a list
		k_neighbor_labels = []  
		for i in k_idx:
			k_neighbor_labels.append(self.y_train[i])
		#k_neighbor_labels = [self.y_train[i] for i in k_idx]  
		
		# return the most common class label
		return(self._most_common_label(k_neighbor_labels))

	def _most_common_label(self, y):
		counter = Counter(y)
		most_common = counter.most_common(1)[0][0]
		return(most_common)


def main():
	print("Hello, this is just a library for kNN classifier")

if __name__ == '__main__':
	main()