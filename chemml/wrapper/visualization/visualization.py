import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

class VisualizeRegression():
	"""
	Module to visualize predictions and compute R^2 score for regression tasks. Use the plot function to create and save an image of the plot.

	Parameters
	----------
	filepath : str, optional, default = "r_square.png"
		Path to store the output plot
	"""

	def __init__(self, filepath='r_square.png'):
		self.filepath = filepath

	def plot(truth, prediction):
		"""
		Method to plot predicted values vs. ground truth and compute R^2 score.

		Parameters
		----------
		truth : array-like
			Ground truth values

		prediction: array-like
			Corresponding predictions
		"""
		f = figure()
		ax = f.add_subplot(111)
		plt.scatter(truth, prediction, 'bx')
		plt.plot(truth, truth, 'r')
		plt.xlabel('Actual Values')
		plt.ylabel('Predicted Values')
		_, _, r2, _, _ = linregress(truth, prediction)
		plt.text(0.6, 0.5, 'R-squared = %0.2f' % r2, transform=ax.transAxes)
		plt.savefig(self.filepath)
