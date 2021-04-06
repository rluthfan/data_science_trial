import numpy as np

class NaiveBayes:
	'''
	Class for Naive Bayes classifier
	'''

	def __init__(self):
		# No initialization in Naive Bayes classifier
		pass

	def fit(self, X, y):
		'''
		Function to calculate class prior and class conditional probability
		'''
		num_samples, num_features = X.shape # Get number of rows/samples and number of features/columns using numpy shape of input
		self._classes = np.unique(y) # Find unique element of output/classes
		num_classes = len(self._classes) # count the number of classes

		# initialize np.zeros to save the conditional probability and prior probability for each class
		self._class_conditionals = np.zeros((num_classes, num_features, 2), dtype=np.float64)
		self._priors =  np.zeros(num_classes, dtype=np.float64)

		# calculate the prior probabilty for each class
		for idx, class_label in enumerate(self._classes):
			X_class = X[y==class_label]
			self._priors[idx] = X_class.shape[0] / float(num_samples) # numpy shape function will give the frequency of the class

			# calculate conditional probability for each feature given the class
			for feat in range(num_features):
				self._class_conditionals[idx, feat, 0] = np.count_nonzero(X_class[:, feat]==0) / X_class.shape[0]
				self._class_conditionals[idx, feat, 1] = np.count_nonzero(X_class[:, feat]) / X_class.shape[0]


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
		Calculate posterior probability given the fitted class conditionals and class priors retrieved from fit() method
		Return back the most likely class
		'''

		# initialize array to save posterior probability
		posteriors = []

		# calculate posterior probability for each class
		for idx, class_label in enumerate(self._classes):
			class_conditional = 1.0
			for feat in range(len(x)):
				class_conditional *= self._class_conditionals[idx, feat, int(bool(x[feat]))]
			prior = (self._priors[idx])
			posterior = class_conditional*prior
			posteriors.append(posterior)
			
		# return class with highest posterior probability
		return(self._classes[np.argmax(posteriors)])

def main():
	print("Hello, this is just a library for Naive Bayes classifier")

if __name__ == '__main__':
	main()