# # Setup

# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

sns.set()

def main(err_plot, runtime_plot):

	# Get data for plotting
	#err_plot = pd.read_excel("saved_value_for_plotting.xlsx")
	seaborn_err = err_plot.melt(id_vars=["depth_values","training_split"],value_vars=err_plot.drop(["depth_values","training_split"],axis=1).columns, var_name="dataset", value_name="error_rate")

	#runtime_plot = pd.read_excel("runtime_plot.xlsx")
	seaborn_rt = runtime_plot.melt(id_vars=["depth_values","training_split"],value_vars=runtime_plot.drop(["depth_values","training_split"],axis=1).columns, var_name="dataset", value_name="runtime_seconds")

	# # Error

	# ## Overall

	plt.figure()
	fig, ax = plt.subplots(figsize=(14, 9))
	sns.lineplot(data=seaborn_err.sort_values(by=["training_split","depth_values","dataset"], ascending=True), x="depth_values", y="error_rate", hue="training_split", style="dataset", palette="flare")
	ax.set(ylim=(-0.05,1.05))
	ax.set_title("Overall Error of different depth with different training size")
	plt.savefig("1-different-split-overall.png", dpi=300)

	# ## Training

	plt.figure()
	fig, ax = plt.subplots(figsize=(14, 9))
	sns.lineplot(data=seaborn_err[(seaborn_err["training_split"]<0.5)&(seaborn_err["dataset"].str.contains("train", regex=True, case=False))], x="depth_values", y="error_rate", hue="training_split", style="dataset", palette="flare")
	ax.set(ylim=(-0.05,1.05))
	ax.set_title("Training set Error of different depth with training size < 0.5")
	plt.savefig("2-1-different-split-training.png", dpi=300)


	plt.figure()
	fig, ax = plt.subplots(figsize=(14, 9))
	sns.lineplot(data=seaborn_err[(seaborn_err["training_split"]>=0.5)&(seaborn_err["dataset"].str.contains("train", regex=True, case=False))], x="depth_values", y="error_rate", hue="training_split", style="dataset", palette="flare")
	ax.set(ylim=(-0.05,1.05))
	ax.set_title("Training set Error of different depth with training size >= 0.5")
	plt.savefig("2-2-different-split-training.png", dpi=300)


	# ## Testing

	plt.figure()
	fig, ax = plt.subplots(figsize=(14, 9))
	sns.lineplot(data=seaborn_err[(seaborn_err["training_split"]<0.5)&(seaborn_err["dataset"].str.contains("test", regex=True, case=False))], x="depth_values", y="error_rate", hue="training_split", style="dataset", palette="flare")
	ax.set(ylim=(-0.05,1.05))
	ax.set_title("Prediction Error of different depth with training size < 0.5")
	plt.savefig("3-1-different-split-predict.png", dpi=300)


	plt.figure()
	fig, ax = plt.subplots(figsize=(14, 9))
	sns.lineplot(data=seaborn_err[(seaborn_err["training_split"]>=0.5)&(seaborn_err["dataset"].str.contains("test", regex=True, case=False))], x="depth_values", y="error_rate", hue="training_split", style="dataset", palette="flare")
	ax.set(ylim=(-0.05,1.05))
	ax.set_title("Prediction Error of different depth with training size >= 0.5")
	plt.savefig("3-2-different-split-predict.png", dpi=300)


	# # Runtime

	# ## Overall

	plt.figure()
	fig, ax = plt.subplots(figsize=(14, 9))
	#sns.lineplot(data=seaborn_rt, x="depth_values", y="runtime_seconds", hue="training_split", style="dataset", palette="crest")
	sns.lineplot(data=seaborn_rt, x="depth_values", y="runtime_seconds", hue="training_split", style="dataset", palette="flare")
	#ax.set(ylim=(-0.05,1.05))
	ax.set_title("Runtime of different depth with different training size")
	plt.savefig("4-runtime-split-overall.png", dpi=300)

	print("Done, the 2 excel files have been visualized to 6 different graphs.")


