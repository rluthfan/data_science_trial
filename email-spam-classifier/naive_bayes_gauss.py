import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class NaiveBayes:
	'''
	Class for Naive Bayes classifier
	Model based on Gaussian distribution
	'''

	def __init__(self):
		pass # no init function for Naive Bayes classifier

	def fit(self, X, y):
		'''
		Function to calculate class prior and class conditional
		Calculate mean, variance for class conditional calculation
		Calculate the class prior probability
		'''
		num_samples, num_features = X.shape # Get number of rows/samples and number of features/columns using numpy shape of input
		self._classes = np.unique(y) # Find unique element of output/classes
		num_classes = len(self._classes) # count the number of classes

		# initialize np.zeros to save the mean, var, and prior for each class
		self._mean = np.zeros((num_classes, num_features), dtype=np.float64)
		self._var = np.zeros((num_classes, num_features), dtype=np.float64)
		self._priors =  np.zeros(num_classes, dtype=np.float64)

		# calculate the mean, var, and prior for each class
		for idx, class_label in enumerate(self._classes):
			X_class = X[y==class_label]
			self._mean[idx, :] = (X_class.astype(np.float64)).mean(axis=0)
			self._var[idx, :] = (X_class.astype(np.float64)).var(axis=0)
			self._priors[idx] = X_class.shape[0] / float(num_samples) # numpy shape function will give the frequency of the class

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
		Calculate class conditional probability using probability density function of Gaussian
		'''

		# initialize array to save posterior probability
		posteriors = []

		# calculate posterior probability for each class
		for idx, class_label in enumerate(self._classes):
			class_conditional = np.sum(np.log(self._pdf(idx, x))) # Use log trick to get the sum of PDF
			prior = np.log(self._priors[idx]) # Get the log of prior probability
			posterior = prior + class_conditional
			posteriors.append(posterior)
			
		# return class with highest posterior probability
		return(self._classes[np.argmax(posteriors)])
			

	def _pdf(self, class_idx, x):
		'''
		Calculate probability density function based on Gaussian distribution
		'''
		mean = self._mean[class_idx]
		var = self._var[class_idx]
		pdf_gauss = (np.exp(- (x-mean)**2 / (2 * var)))/(np.sqrt(2 * np.pi * var))
		return(pdf_gauss)

def main():
	print("Hello, this is just a library for Naive Bayes classifier")

if __name__ == '__main__':
	main()