# k-means and kernelization object
from kmeans_scratch import kMeans
from kmeans_kernel import kernelize_dataset

# standard machine learning library
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
sns.set()

# Randomly generating datatype

n_samples = 500

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
# no_structure = np.random.rand(n_samples, 2), None

X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)


datasets = [(blobs, 3), (noisy_circles, 2), (noisy_moons, 2), (aniso,3)]

# Part 3c1
fig, axs = plt.subplots(len(datasets),2, figsize=(5*2, 3*len(datasets)))
for i in range(1,len(datasets)):
	X,y = datasets[i][0]

	# normalize dataset for easier parameter selection
	# X = StandardScaler().fit_transform(X)

	clustering = kMeans(k=datasets[i][1], max_iter=150)
	clustering.fit(X)

	y_pred = clustering.predict(X)
	centers = clustering.centroids
	
	sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette="tab10", ax=axs[i,0])
	axs[i,0].set_title("True")
	sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_pred, palette="tab10", ax=axs[i,1])
	sns.scatterplot(x=centers.T[0], y=centers.T[1], color="red", s=100, ax=axs[i,1])
	axs[i,1].set_title("Prediction")

plt.savefig("3c1-1.png", dpi=300)

# Part 3c6
for i in range(len(datasets)):
	X,y = datasets[i][0]
	
	# normalize dataset for easier parameter selection
	X = StandardScaler().fit_transform(X)

	dfs = []
	# test for multiple neighbors
	for neighbor in range(1,12+1):
		kernel = kernelize_dataset(k=datasets[i][1],r=neighbor)
		X_transformed = kernel.transform(X)

		clustering = kMeans(k=datasets[i][1], max_iter=150)
		clustering.fit(X_transformed)

		y_pred = clustering.predict(X_transformed)
		centers = clustering.centroids

		dict_res = {"r_neighbors": neighbor,"plot_x_axis":X[:,0], "plot_y_axis":X[:,1], "y_pred": y_pred}
		df_plot = pd.DataFrame(dict_res)
		dfs.append(df_plot)

	df_result = pd.concat(dfs).reset_index(drop=True)
	g = sns.FacetGrid(df_result, col="r_neighbors", hue="y_pred", palette="tab10", col_wrap=3)
	g.map(sns.scatterplot, "plot_x_axis", "plot_y_axis")
	plt.savefig("3c6-{}.png".format(i), dpi=300)



