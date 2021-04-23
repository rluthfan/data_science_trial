import numpy as np

def euclidean_distance(features, predict, order=2):
	'''
	Calculate vector norm
	'''
	return(np.linalg.norm(np.array(features)-np.array(predict), ord=order))

class kMeans:
	'''
	Class for k-Means clustering
	'''

	def __init__(self, k=3, max_iter=100):
		self.k = k # number of clusters the algorithm will form
		self.max_iter = max_iter # max number of iterations the algorithm will run to ensure convergence

	def fit(self, X):
		'''
		Initialize initial centroids
		'''

		self.X_train = X
		self.n_samples, self.n_features = X.shape

		# Initialize centroids
		random_idxs = np.random.choice(self.n_samples, self.k, replace=False)
		centroids = [self.X_train[i] for i in random_idxs]

		self.centroids = np.array(centroids)


	def predict(self, X):
		'''
		Do K-Means clustering and return cluster indices
		'''

		# Iteratively optimize clusters until convergence or max iterations is reached
		for _ in range(self.max_iter):

			# Get new centroids of the created clusters
			new_centroid = np.array([np.mean(X[i],axis=0) for i in self._create_clusters(X)])
			
			# Compare new and old centroid, update centroid when there is any difference, break when it has achieve convergence
			if (new_centroid - self.centroids).any():
				self.centroids = new_centroid
			else:
				break

		return(np.array([self._closest_centroid(x) for x in X]))

	def _create_clusters(self, X):
		'''
		Function to generate clusters based on closest centroids
		'''

		# Assign the samples to the closest centroids to create clusters
		pred_res = np.array([self._closest_centroid(x) for x in X])

		# Group index of X by cluster
		res = [((pred_res == i).nonzero()[0]) for i in np.unique(pred_res)]

		# Return grouped indices
		return(res)

	def _closest_centroid(self, x):
		'''
		Get closest centroids to data point
		'''

		# Compute distances between sample x and each centroids
		distances = [euclidean_distance(centroid, x) for centroid in self.centroids]
		
		# Sort by distance and return index of the closest centroid
		k_idx = np.argmin(distances)
		
		# return the closest centroid
		return(k_idx)


def main():
	print("Hello, this is just a library for k-means clustering")

if __name__ == '__main__':
	main()