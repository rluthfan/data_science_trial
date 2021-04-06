import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn')
import seaborn as sns

from sklearn.model_selection import train_test_split

# Preprocessing function
import a_preproc_func

# Algorithm designed from scratch
import kNN_scratch
import naive_bayes_scratch
import tree_decision_scratch

import time

def evaluate(y_true, y_pred):
	'''
	Calculate the accuracy of classifier
	'''
	accuracy = np.sum(y_true == y_pred) / len(y_true)
	return(accuracy)

def search_param(X_train, X_validation, y_train, y_validation, param, param_values):
	accuracy = 0.0
	accuracies = []
	best_param = 1
	for i in param_values:
		if param == "k":
			clf = kNN_scratch.KNN(k=i)
			clf.fit(X_train, y_train)
			temp_acc = evaluate(y_validation, clf.predict(X_validation))
			accuracies.append(temp_acc)
			if temp_acc > accuracy:
				accuracy = temp_acc
				best_param = i
		elif param == "l_norm_used":
			clf = kNN_scratch.KNN(l_norm_used=i)
			clf.fit(X_train, y_train)
			temp_acc = evaluate(y_validation, clf.predict(X_validation))
			accuracies.append(temp_acc)
			if temp_acc > accuracy:
				accuracy = temp_acc
				best_param = i
		elif param == "min_samples_split":
			clf = tree_decision_scratch.DecisionTree(min_samples_split=i)
			clf.fit(X_train, y_train)
			temp_acc = evaluate(y_validation, clf.predict(X_validation))
			accuracies.append(temp_acc)
			if temp_acc > accuracy:
				accuracy = temp_acc
				best_param = i
		elif param == "max_depth":
			clf = tree_decision_scratch.DecisionTree(max_depth=i)
			clf.fit(X_train, y_train)
			temp_acc = evaluate(y_validation, clf.predict(X_validation))
			accuracies.append(temp_acc)
			if temp_acc > accuracy:
				accuracy = temp_acc
				best_param = i
	
	plt.figure()
	if param == "l_norm_used":
		ax = sns.lineplot(x=[1,2,3], y=accuracies, marker="o")
	else: ax = sns.lineplot(x=param_values, y=accuracies, marker="o")
	ax.set(ylim=(0.7,1.0))
	ax.set_title('Accuracies of different '+ param +' values')
	plt.savefig('different-' + param + '-values.png', dpi=300)
	plt.clf()
	return(best_param)


def main():
	
	# load dataset with file directory and maximum number of features
	X, y = a_preproc_func.main(fd="./enron1", max_feats=100)
	#X,y = datasets.load_breast_cancer(return_X_y=True)

	# split training, validation, test datasets
	test_size = 0.2
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)
	valid_size = 0.3
	X_training, X_validation, y_training, y_validation = train_test_split(X, y, test_size=valid_size, random_state=101)

	# Using default values for first run cycle
	k = 3
	p = 2
	clf = kNN_scratch.KNN(k=k, l_norm_used=p)
	start_time = time.time()
	clf.fit(X_training, y_training)
	print("kNN Training Time")
	print("--- %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	print("Custom kNN classification accuracy validation set = {:.2f}".format(evaluate(y_validation, clf.predict(X_validation))))
	print("kNN Prediction Time")
	print("--- %s seconds ---" % (time.time() - start_time))

	
	clf = naive_bayes_scratch.NaiveBayes()
	start_time = time.time()
	clf.fit(X_training, y_training)
	print("Naive Bayes Training Time")
	print("--- %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	print("Custom Naive Bayes classification accuracy validation set = {:.2f}".format(evaluate(y_validation, clf.predict(X_validation))))
	print("Naive Bayes Prediction Time")
	print("--- %s seconds ---" % (time.time() - start_time))
	
	
	depth = 10
	sample_min = 2
	clf = tree_decision_scratch.DecisionTree(min_samples_split=sample_min, max_depth=depth)
	start_time = time.time()
	clf.fit(X_training, y_training)
	print("Decision Tree Training Time")
	print("--- %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	print("Custom Decision Tree classification accuracy validation set = {:.2f}".format(evaluate(y_validation, clf.predict(X_validation))))
	print("Decision Tree Prediction Time")
	print("--- %s seconds ---" % (time.time() - start_time))
	
	# Get best hyperparameters
	start_time = time.time()
	k_values = [i for i in range(1,10)]
	best_k = search_param(X_training, X_validation, y_training, y_validation, "k", k_values)

	l_norm_values = [1,2,np.inf]
	best_l_norm = search_param(X_training, X_validation, y_training, y_validation, "l_norm_used", l_norm_values)

	split_values = [i for i in range(2,11)]
	best_min_split = search_param(X_training, X_validation, y_training, y_validation, "min_samples_split", split_values)

	depth_values = [i for i in range(1,21)]
	best_max_depth = search_param(X_training, X_validation, y_training, y_validation, "max_depth", depth_values)

	print("Hyperparameter tuning time")
	print("--- %s seconds ---" % (time.time() - start_time))

	# Use best hyperparameter for test set predictions with different sample split size
	test_sizes = np.arange(0.1, 0.55, 0.05)
	knn_accuracies = []
	naive_bayes_accuracies = []
	decision_tree_accuracies = []

	start_time = time.time()
	for i in test_sizes:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=101)

		clf = kNN_scratch.KNN(k=best_k, l_norm_used=best_l_norm)
		clf.fit(X_train, y_train)
		knn_accuracies.append(evaluate(y_test, clf.predict(X_test)))

		clf = naive_bayes_scratch.NaiveBayes()
		clf.fit(X_train, y_train)
		naive_bayes_accuracies.append(evaluate(y_test, clf.predict(X_test)))
		
		clf = tree_decision_scratch.DecisionTree(min_samples_split=best_min_split, max_depth=best_max_depth)
		clf.fit(X_train, y_train)
		decision_tree_accuracies.append(evaluate(y_test, clf.predict(X_test)))

	print("Runtime")
	print("--- %s seconds ---" % (time.time() - start_time))

	plt.figure()
	f, ax = plt.subplots(1, 1)
	sns.lineplot(x=test_sizes, y=knn_accuracies, label = "kNN accuracies", color="blue", marker="o")
	sns.lineplot(x=test_sizes, y=naive_bayes_accuracies, label = "Naive Bayes accuracies", color="red", marker="o")
	sns.lineplot(x=test_sizes, y=decision_tree_accuracies, label = "Decision Tree accuracies", color="green", marker="o")
	ax.set(ylim=(0.5,1.0))
	ax.set_title('Accuracies of different test sizes')
	plt.legend()
	plt.savefig('different-test-sizes.png', dpi=300)
	plt.clf()

if __name__ == '__main__':
	main()
