# src: http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

from skimage.measure import compare_ssim as ssim #structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()

# load the images -- the original, the original + contrast,
# and the original + photoshop
original1 = cv2.imread("images2compare/dcgan_generated_image_epoch_1000_1000.png")
original2 = cv2.imread("images2compare/gan_generated_image_epoch_1000_1000.png")
#contrast = cv2.imread("images2compare/jp_gates_contrast.png")
#shopped = cv2.imread("images2compare/jp_gates_photoshopped.png")
 
# convert the images to grayscale
original1 = cv2.cvtColor(original1, cv2.COLOR_BGR2GRAY)
original2 = cv2.cvtColor(original2, cv2.COLOR_BGR2GRAY)
#contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
#shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Original1", original1), ("Original2", original2) #, ("Contrast", contrast), ("Photoshopped", shopped)
 
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
 
# show the figure
plt.show()
 
# compare the images
compare_images(original1, original2, "Original1 vs. Original2")
#compare_images(original, contrast, "Original vs. Contrast")
#compare_images(original, shopped, "Original vs. Photoshopped")
