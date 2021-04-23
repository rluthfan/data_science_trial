import numpy as np
from sklearn.neighbors import NearestNeighbors

class kernelize_dataset:
	'''
	Class for kernelizing dataset before using k-means clustering
	'''

	def __init__(self, k=3, r=3, algo="auto"):
		self.k = k # number of cluster to be used as dimension
		self.r = r # number of nearest neighbors
		self.algo = algo # algorithm for finding the nearest neighbor

	def fit(self, X):
		'''
		Initialize initial nearest neighbor using sklearn neighbors
		'''

		self.X = X
		self.n_samples, self.n_features = X.shape

		# Initialize nearest neighbors matrix
		nbrs = NearestNeighbors(n_neighbors=self.r, algorithm=self.algo).fit(X)
		distances, indices = nbrs.kneighbors(X)
		nbrs_mat = nbrs.kneighbors_graph(X).toarray()

		# Change diagonal to become 0 as opposed to 1
		np.fill_diagonal(nbrs_mat, np.zeros(self.n_samples,dtype=np.float64))

		# Make element[i,j] = element[j,i] by getting all the neighbors using max
		nbrs_max = np.maximum(nbrs_mat, nbrs_mat.T)

		# Get diagonal matrix of sum neighbor weight vector
		diag_m = np.diag(np.sum(nbrs_max, axis=1))

		# Get transformation matrix W
		w_m = diag_m - nbrs_max

		self.transform_matrix = w_m


	def transform(self, X):
		'''
		Return the transformed data
		'''
		
		self.fit(X)

		# Get bottom k eigenvectors/values
		eigenValues, eigenVectors = np.linalg.eig(self.transform_matrix)
		idx = eigenValues.argsort()   
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]

		return(eigenVectors[:,:self.k])


def main():
	print("Hello, this is just a library for kernelizing dataset before using k-means")

if __name__ == '__main__':
	main()