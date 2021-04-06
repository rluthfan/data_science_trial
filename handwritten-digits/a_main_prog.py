import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn')
import seaborn as sns

import scipy.io

from sklearn.model_selection import train_test_split
#import sklearn.datasets as datasets

# Algorithm designed from scratch
import tree_decision_scratch

# Custom plotting function
import a_plot_func

import time

def load_dataset(fd, max_feats):
	'''
	Load .mat data using scipy.io
	'''
	dotmat_data = scipy.io.loadmat(fd)
	image_features = dotmat_data['X']
	image_labels = dotmat_data['Y'][:,0]

	# Choose pixel with highest variance to reduce maximum number of features
	variance = np.var(image_features, axis=0)
	pixel_used = np.argsort(variance)[-max_feats::]
	image_features = np.take(image_features, pixel_used, axis=1)

	return (image_features, image_labels)

def evaluate(y_true, y_pred):
	'''
	Calculate the accuracy of classifier
	'''
	accuracy = np.sum(y_true == y_pred) / len(y_true)
	return(accuracy)

def main():
	
	# load dataset with file directory and maximum number of features
	X, y = load_dataset(fd="./digits.mat", max_feats=200)
	#X, y = datasets.load_digits(return_X_y=True)

	# Test set predictions with different sample split size
	train_sizes = np.arange(0.10, 0.95, 0.10)

	# minimum number of sample in each node for decision tree
	sample_min = 2
	# Use hyperparameter max depth for test set predictions with different value
	depth_values = np.append(np.arange(1, 5, 1), np.arange(5, 41, 5))

	# Initialize dictionary for saving data values
	dict_error = {"depth_values": [], "training_split": [], "train_error": [], "test_error": []}
	dict_runtime = {"depth_values": [], "training_split": [], "train_runtime": [], "predict_runtime": []}

	for split in train_sizes:
		# Create random split using sklearn train test split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1.0-split), stratify=y) #(, random_state=101)

		for depth in depth_values:
			
			# Create classifier object with different depth value and Training split in each iterations
			clf = tree_decision_scratch.DecisionTree(min_samples_split=sample_min, max_depth=depth)

			# Training process -> fitting features and label to build tree
			start_time = time.time()
			clf.fit(X_train, y_train)
			train_rt = (time.time() - start_time)

			# Prediction error rate for training data
			tr_err = 1.0 - evaluate(y_train, clf.predict(X_train))
			# Prediction error rate for test data
			start_time = time.time()
			te_err = 1.0 - evaluate(y_test, clf.predict(X_test))
			pred_rt = (time.time() - start_time)

			# Append values to dictionaries
			dict_error["training_split"].append(split)
			dict_error["depth_values"].append(depth)
			dict_error["train_error"].append(tr_err)
			dict_error["test_error"].append(te_err)

			dict_runtime["training_split"].append(split)
			dict_runtime["depth_values"].append(depth)
			dict_runtime["train_runtime"].append(train_rt)
			dict_runtime["predict_runtime"].append(pred_rt)
	
	# Output dictionary to excel using pandas library
	error_plot = pd.DataFrame(dict_error)
	error_plot.to_excel("saved_value_for_plotting.xlsx", index=False) 

	runtime_plot = pd.DataFrame(dict_runtime)
	runtime_plot.to_excel("runtime_plot.xlsx", index=False) 

	print("Done, observations saved to output 2 excel files.")

	# Plot the error dataset and runtime
	a_plot_func.main(err_plot=error_plot, runtime_plot=runtime_plot)


if __name__ == '__main__':
	main()
