
from skimage.feature import plot_matches
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os

def show_correspondences(imgA, imgB, X1, Y1, X2, Y2, matches, good_matches, number_to_display, filename=None):
	'''
		Visualizes corresponding points between two images, either as
		arrows or dots

		mode='dots': Corresponding points will have the same random color
		mode='arrows': Corresponding points will be joined by a line

		Writes out a png of the visualization if 'filename' is not None.
	'''

	# generates unique figures so students can
	# look at all three at once
	fig, ax = plt.subplots(nrows=1, ncols=1)

	matches = matches[0:number_to_display, :]
	good_matches = good_matches[0:number_to_display]


	kp1 = zip_x_y(Y1, X1)
	kp2 = zip_x_y(Y2, X2)
	matches = matches.astype(int)
	plot_matches(ax, imgA, imgB, kp1, kp2, matches[np.logical_not(good_matches)], matches_color='orangered')
	plot_matches(ax, imgA, imgB, kp1, kp2, matches[good_matches], matches_color='springgreen')

	fig = plt.gcf()
	plt.show()

	if filename:
		if not os.path.isdir('../results'):
			os.mkdir('../results')
		fig.savefig('../results/' + filename)

	return

def show_correspondences_custom_image(imgA, imgB, X1, Y1, X2, Y2, matches, scale_factor, number_to_display, filename=None):
	'''
		Visualizes corresponding points between two images, either as
		arrows or dots. Unlike show_correspondences, does not take correct_matches argument

		mode='dots': Corresponding points will have the same random color
		mode='arrows': Corresponding points will be joined by a line

		Writes out a png of the visualization if 'filename' is not None.
	'''

	# generates unique figures so students can
	# look at all three at once
	fig, ax = plt.subplots(nrows=1, ncols=1)

	matches = matches[0:number_to_display, :]

	x1_scaled = X1 / scale_factor
	y1_scaled = Y1 / scale_factor
	x2_scaled = X2 / scale_factor
	y2_scaled = Y2 / scale_factor

	kp1 = zip_x_y(y1_scaled, x1_scaled)
	kp2 = zip_x_y(y2_scaled, x2_scaled)
	matches = matches.astype(int)
	plot_matches(ax, imgA, imgB, kp1, kp2, matches, matches_color='yellow')

	fig = plt.gcf()
	plt.show()

	if filename:
		if not os.path.isdir('../results'):
			os.mkdir('../results')
		fig.savefig('../results/' + filename)

	return

def zip_x_y(x, y):
	zipped_points = []
	for i in range(len(x)):
		zipped_points.append(np.array([x[i], y[i]]))
	return np.array(zipped_points)
