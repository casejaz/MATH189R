import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import copy

img = Image.open("/Users/yvenica/Desktop/MATH189R/HW2/problem 5/crown.jpg")

imgM = np.array(list(img.getdata(band=0)), float)


imgM.shape = (img.size[1], img.size[0])
imgM = np.matrix(imgM)

shuffledImgM = copy.deepcopy(imgM)
shape0 = shuffledImgM.shape[0]
shape1 = shuffledImgM.shape[1]
newShuffledList = []
for i in range(shape0 * shape1):
	newShuffledList.append(shuffledImgM.item(i))

np.random.shuffle(newShuffledList)
shuffledImgM = np.matrix(newShuffledList)
shuffledImgM = np.reshape(shuffledImgM, (shape0,shape1))
U, sigma, V = np.linalg.svd(imgM)
US, sigmaS, VS = np.linalg.svd(shuffledImgM)


def plotGen():
	plt.figure(1)
	plt.style.use('ggplot')
	orderedPlot, = plt.plot(range(100), sigma[:100], 'b')
	plt.title('Progression of the 100 Largest Singular Values')

	shuffledSigma = copy.deepcopy(sigma)
	random.shuffle(shuffledSigma)
	shuffledPlot, = plt.plot(range(100), sigmaS[:100], 'ro')
	plt.legend((orderedPlot, shuffledPlot), ("original","random"), loc=1)
	plt.show()

def gridGen():
	U, sigma, V = np.linalg.svd(imgM)
	kArr = [2, 10, 20]
	reconstimg = np.matrix(U[:,:1] * np.diag(sigma[:1]) * np.matrix(V[:1, :]))
	plt.figure(1)
	plt.subplot(221)
	plt.imshow(imgM, cmap='gray')
	plt.title('Original')

	plt.subplot(222)
	reconstimg = np.matrix(U[:,:2] * np.diag(sigma[:2]) * np.matrix(V[:2, :]))
	plt.imshow(reconstimg, cmap='gray')
	plt.title('k = 2')

	plt.subplot(223)
	reconstimg = np.matrix(U[:,:10] * np.diag(sigma[:10]) * np.matrix(V[:10, :]))
	plt.imshow(reconstimg, cmap='gray')
	plt.title('k = 10')

	plt.subplot(224)
	reconstimg = np.matrix(U[:,:20] * np.diag(sigma[:20]) * np.matrix(V[:20, :]))
	plt.imshow(reconstimg, cmap='gray')
	plt.title('k = 20')

	plt.show()

plotGen()
gridGen()