#-*-coding:utf-8-*-

import matplotlib.pyplot as plt
import numpy as np
import cv2, csv, sys, os

# from skimage.color import rgb2gray
# from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

filename = 'homework_fcc.bmp'
k_size = 7
#filename = 'particle.jpg'
#filename = 'proteasome_0001.tif'
#filename = 'Ni_NP_Gr_DW.jpeg'
#k_size = 125
#k_size = 15
# filename = sys.argv[1]

def set_init(x0, xr, y0, yr):

	npt = 2*np.max([xr,yr])
	s = np.linspace(0, 2*np.pi, npt )
	x = x0 + xr*np.cos(s)
	y = y0 + yr*np.sin(s)
	init = np.array([x, y]).T

	return init


def mat_max(mat, k_center):

	ind = np.unravel_index(np.argmax(mat, axis = None), mat.shape)
	if ind == (k_center,k_center):
		mat_center_max = True
	else:
		mat_center_max = False

	return mat_center_max


def find_peak(img,k_size):

	size_x = img.shape[0]
	size_y = img.shape[1]
	print img.shape
	peakind = []
	mat = np.zeros([k_size,k_size])
	k_center = int(k_size/2)
	for i in range(k_center,size_x-k_center) :
		for j in range(k_center,size_y-k_center):
			mat = img[i-k_center:i+k_center+1,j-k_center:j+k_center+1]
			if mat_max(mat,k_center) == True:
				peakind.append([j, img[i][j]*2, i, img[i][j]*2])

	return peakind


def main():

	img = cv2.imread(filename)
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	## Binary (OTSU's method)
	ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	#ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	print 'Threshold (pixel) : ', ret

	## Remove noise
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

	## Distance transformation
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L1,3)
#	dist_transform = cv2.distanceTransform(opening,cv2.DIST_C,3)
	dist_transform_8u = np.uint8(dist_transform)

	## Find particles
	particles = find_peak(dist_transform_8u,k_size)
	print particles
	print 'Number of particles : ', len(particles)

	## Shape discription (Snake algorithm)
	init_list = []
	init_list2 = []
	snake_list = []
	particle_list = []
	k = 0
	for p in particles:
		k+=1
		print k
		init = set_init(p[0], p[1], p[2], p[3])
		init_list.append(init), init_list2.extend(init)
		contour = (active_contour(gaussian(thresh, 3),
		                       init, alpha=0.015, beta=10, gamma=0.001))
		snake_list.append(contour)
		particle_list.extend(contour)
	print np.array(snake_list)
	print cv2.contourArea(np.array(snake_list))
	with open("snake_list_%s.csv" % filename.split('.')[0], "wb") as f:
		writer = csv.writer(f)
		writer.writerows(snake_list)
	f.close()

	## Plot figures
	plt.figure(),plt.imshow(img),plt.title('Original'),plt.xticks([]),plt.yticks([])
	plt.savefig("figures_%s_%s.png" % (filename.split('.')[0], 'Original'))

	plt.figure(),plt.imshow(imgray),plt.title('Gray'),plt.xticks([]),plt.yticks([])
	plt.savefig("figures_%s_%s.png" % (filename.split('.')[0], 'Gray'))

	plt.figure(),plt.imshow(thresh),plt.title('Binary'),plt.xticks([]),plt.yticks([])
	plt.savefig("figures_%s_%s.png" % (filename.split('.')[0], 'Binary'))

	plt.figure(),plt.imshow(dist_transform_8u, cmap=plt.cm.gray), plt.title('Distance transform'),plt.xticks([]),plt.yticks([])
	plt.savefig("figures_%s_%s.png" %(filename.split('.')[0], 'Distance'))

	plt.figure(), plt.hist(imgray.ravel(), 256, color='black'), plt.title('Histogram'), plt.xticks([]), plt.yticks([])
	plt.savefig("figures_%s_%s.png" %(filename.split('.')[0], 'Histogram'))

	plt.close()

	plt.figure(),plt.imshow(thresh), plt.title('Binary+Snake'),plt.xticks([]),plt.yticks([])
	for p in range(len(particles)):
		#plt.plot(particles[p][0], particles[p][2], 'og')#, lw=3)
		plt.plot(init_list[p][:, 0], init_list[p][:, 1], '--r')#, lw=3)
		plt.plot(snake_list[p][:, 0], snake_list[p][:, 1], '-b')#, lw=3)
	plt.savefig("figures_%s_Binary_Snake.png" % filename.split('.')[0])
	plt.close()

	plt.figure(),plt.imshow(dist_transform_8u, cmap=plt.cm.gray), plt.title('Distance+Snake'),plt.xticks([]),plt.yticks([])
	for p in range(len(particles)):
		#plt.plot(particles[p][0], particles[p][2], 'og')#, lw=3)
		plt.plot(init_list[p][:, 0], init_list[p][:, 1], '--r')#, lw=3)
		plt.plot(snake_list[p][:, 0], snake_list[p][:, 1], '-b')#, lw=3)
	plt.savefig("figures_%s_Distance_Snake.png" % filename.split('.')[0])
	plt.close()

	plt.figure(),plt.imshow(img), plt.title('Original+Snake'),plt.xticks([]),plt.yticks([])
	for p in range(len(particles)):
		#plt.plot(particles[p][0], particles[p][2], 'og')#, lw=3)
		#plt.plot(init_list[p][:, 0], init_list[p][:, 1], '--r')#, lw=3)
		plt.plot(snake_list[p][:, 0], snake_list[p][:, 1], '-b')#, lw=3)
	plt.savefig("figures_%s_Original_Snake.png" % filename.split('.')[0])
	plt.close()
	# plt.show()


if __name__ == '__main__':
	main()
