from django.shortcuts import render
from .models import Input

import shutil

import math
import cv2
from skimage.color import rgb2hsv

import datetime
import os
import math
import numpy as np
import cv2
from PIL import Image
import os
import cv2
import natsort
import xlwt
import cv2
from skimage import exposure

from django.conf import settings
from django.conf.urls.static import static

from .CLAHE.sceneRadianceCLAHE import RecoverCLAHE
from .CLAHE.sceneRadianceHE import RecoverHE

from .RAY.color_equalisation import RGB_equalisation
from .RAY.global_stretching_RGB import stretching
from .RAY.hsvStretching import HSVStretching

from .RAY.histogramDistributionLower import histogramStretching_Lower
from .RAY.histogramDistributionUpper import histogramStretching_Upper
from .RAY.rayleighDistribution import rayleighStretching
from .RAY.rayleighDistributionLower import rayleighStretching_Lower
from .RAY.rayleighDistributionUpper import rayleighStretching_Upper
from .RAY.sceneRadiance import sceneRadianceRGB

from .DCP.GuidedFilter import GuidedFilter

from .MIP.BL import getAtomsphericLight
from .MIP.EstimateDepth import DepthMap
from .MIP.getRefinedTramsmission import Refinedtransmission
from .MIP.TM import getTransmission
from .MIP.sceneRadiance import sceneRadianceRGBMIP

def index(request):
    return render(request,'index.html')

def clahe(request):
	return render(request,'clahe.html')

def rayleigh(request):
	return render(request,'rayleigh.html')

def mip(request):
	return render(request,'mip.html')
def dcp(request):
	return render(request,'dcp.html')

def both(request):
	return render(request,'both.html')

def auto(request):
	return render(request,'auto.html')

def get_image(request):
	print('get_image')
	shutil.rmtree("UWIE/static/Input/")
	if request.method == "POST":
		in_img = request.FILES['image']
		in_img.name = "input.jpg"
		input = Input(img = in_img)
		input.save()
		enhanceImageCLAHE("UWIE/static")
		img1 = in_img
		img2 = "CLAHE.jpg"
	return render(request,'clahe.html',{'img1':img1,'img2':img2})

def enhanceImageCLAHE(folder):
	np.seterr(over='ignore')
	path = folder + "/Input"
	files = os.listdir(path)
	files =  natsort.natsorted(files)
	for i in range(len(files)):
		file = files[i]
		filepath = path + "/" + file
		prefix = file.split('.')[0]
		if os.path.isfile(filepath):
			print('********    file   ********',file)
			img = cv2.imread(folder + '/Input/' + file)
			sceneRadiance = RecoverCLAHE(img)
			cv2.imwrite(folder + '/Input/' + 'CLAHE.jpg', sceneRadiance)

def get_image_ray(request):
	print('get_imageray')
	shutil.rmtree("UWIE/static/Input/")
	if request.method == "POST":
		in_img = request.FILES['image']
		in_img.name = "input.jpg"
		input = Input(img = in_img)
		input.save()
		enhanceImageRAY("UWIE/static")
		img1 = in_img
		img2 = "RAY.jpg"
	return render(request,'rayleigh.html',{'img1':img1,'img2':img2})

def get_image_mip(request):
	print('get_image_mip')
	shutil.rmtree("UWIE/static/Input/")
	if request.method == "POST":
		in_img = request.FILES['image']
		in_img.name = "input.jpg"
		input = Input(img = in_img)
		input.save()
		restoreMIP("UWIE/static")
		img1 = in_img
		img2 = "MIP.jpg"
	return render(request,'mip.html',{'img1':img1,'img2':img2})

def get_image_dcp(request):
	print('get_image_mip')
	shutil.rmtree("UWIE/static/Input/")
	if request.method == "POST":
		in_img = request.FILES['image']
		in_img.name = "input.jpg"
		input = Input(img = in_img)
		input.save()
		restoreDCP("UWIE/static")
		img1 = in_img
		img2 = "MIP.jpg"
	return render(request,'mip.html',{'img1':img1,'img2':img2})

def enhanceImageRAY(folder):
	e = np.e
	esp = 2.2204e-16
	np.seterr(over='ignore')
	if __name__ == '__main__':
	    pass

	path = folder + "/Input"
	files = os.listdir(path)
	files =  natsort.natsorted(files)

	for i in range(len(files)):
	    file = files[i]
	    filepath = path + "/" + file
	    prefix = file.split('.')[0]
	    if os.path.isfile(filepath):
	        print('********    file   ********',file)
	        img = cv2.imread(folder + '/Input/' + file)
	        prefix = file.split('.')[0]
	        height = len(img)
	        width = len(img[0])

	        sceneRadiance = RGB_equalisation(img, height, width)

	        sceneRadiance = stretching(sceneRadiance)
	        sceneRadiance_Lower, sceneRadiance_Upper = rayleighStretching(sceneRadiance, height, width)

	        sceneRadiance = (np.float64(sceneRadiance_Lower) + np.float64(sceneRadiance_Upper)) / 2

	        sceneRadiance = HSVStretching(sceneRadiance)
	        sceneRadiance = sceneRadianceRGB(sceneRadiance)
	        cv2.imwrite(folder + '/Input/' + 'RAY.jpg', sceneRadiance)

def get_image_both(request):
	print('get_imageray')
	shutil.rmtree("UWIE/static/Input/")
	if request.method == "POST":
		in_img = request.FILES['image']
		in_img.name = "input.jpg"
		input = Input(img = in_img)
		input.save()
		enhanceImageCLAHE("UWIE/static")
		enhanceImageRAY("UWIE/static")
		img1 = in_img
		img2 = "RAY.jpg"
		img3 = "CLAHE.jpg"
	return render(request,'both.html',{'img1':img1,'img2':img2,'img3':img3})

def autogetimage(request):
	shutil.rmtree("UWIE/static/Input/")
	if request.method == "POST":
		in_img = request.FILES['image']
		in_img.name = "input.jpg"
		input = Input(img = in_img)
		input.save()
		enhanceImageCLAHE("UWIE/static")
		enhanceImageRAY("UWIE/static")
		restoreDCP("UWIE/static")
		restoreMIP("UWIE/static")
		img1 = in_img
		img2 = "RAY.jpg"
		img3 = "CLAHE.jpg"
		img4 = "DCP.jpg"
		img5 = "MIP.jpg"
		img6 = "DCP_TM.jpg"
		img7 = "MIP_TM.jpg"
	return render(request,'auto.html',
		{'img1':img1,'img2':img2,'img3':img3,'img4':img4,'img5':img5,'img6':img6,'img7':img7})

def restoreDCP(folder):
	class Node(object):
	    def __init__(self, x, y, value):
	        self.x = x
	        self.y = y
	        self.value = value

	    def printInfo(self):
	        print(self.x, self.y, self.value)


	def getMinChannel(img):
	    
	    if len(img.shape) == 3 and img.shape[2] == 3:
	        pass
	    else:
	        print("bad image shape, input must be color image")
	        return None
	    imgGray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
	    localMin = 255

	    for i in range(0, img.shape[0]):
	        for j in range(0, img.shape[1]):
	            localMin = 255
	            for k in range(0, 3):
	                if img.item((i, j, k)) < localMin:
	                    localMin = img.item((i, j, k))
	            imgGray[i, j] = localMin
	    return imgGray



	def getDarkChannel(img, blockSize):
	   
	    if len(img.shape) == 2:
	        pass
	    else:
	        print("bad image shape, input image must be two demensions")
	        return None


	    if blockSize % 2 == 0 or blockSize < 3:
	        print('blockSize is not odd or too small')
	        return None

	    addSize = int((blockSize - 1) / 2)
	    newHeight = img.shape[0] + blockSize - 1
	    newWidth = img.shape[1] + blockSize - 1

	    imgMiddle = np.zeros((newHeight, newWidth))
	    imgMiddle[:, :] = 255
	   
	    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img

	    imgDark = np.zeros((img.shape[0], img.shape[1]), np.uint8)
	    localMin = 255

	    for i in range(addSize, newHeight - addSize):
	        for j in range(addSize, newWidth - addSize):
	            localMin = 255
	            for k in range(i - addSize, i + addSize + 1):
	                for l in range(j - addSize, j + addSize + 1):
	                    if imgMiddle.item((k, l)) < localMin:
	                        localMin = imgMiddle.item((k, l))
	            imgDark[i - addSize, j - addSize] = localMin

	    return imgDark



	def getAtomsphericLight(darkChannel, img, meanMode=False, percent=0.001):
	    size = darkChannel.shape[0] * darkChannel.shape[1]
	    height = darkChannel.shape[0]
	    width = darkChannel.shape[1]

	    nodes = []

	    for i in range(0, height):
	        for j in range(0, width):
	            oneNode = Node(i, j, darkChannel[i, j])
	            nodes.append(oneNode)


	    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

	    atomsphericLight = 0


	    if int(percent * size) == 0:
	        for i in range(0, 3):
	            if img[nodes[0].x, nodes[0].y, i] > atomsphericLight:
	                atomsphericLight = img[nodes[0].x, nodes[0].y, i]
	        return atomsphericLight

	    if meanMode:
	        sum = 0
	        for i in range(0, int(percent * size)):
	            for j in range(0, 3):
	                sum = sum + img[nodes[i].x, nodes[i].y, j]

	        atomsphericLight = int(sum / (int(percent * size) * 3))
	        return atomsphericLight


	    for i in range(0, int(percent * size)):
	        for j in range(0, 3):
	            if img[nodes[i].x, nodes[i].y, j] > atomsphericLight:
	                atomsphericLight = img[nodes[i].x, nodes[i].y, j]

	    return atomsphericLight

	def getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001):
	    
	    gimfiltR = 50 
	    eps = 10 ** -3  
	    
	    imgGray = getMinChannel(img)

	    imgDark = getDarkChannel(imgGray, blockSize=blockSize)
	    atomsphericLight = getAtomsphericLight(imgDark, img, meanMode=meanMode, percent=percent)

	    imgDark = np.float64(imgDark)
	    transmission = 1 - omega * imgDark / atomsphericLight

	    guided_filter = GuidedFilter(img, gimfiltR, eps)
	    transmission = guided_filter.filter(transmission)
	    


	    transmission = np.clip(transmission, t0, 0.9)

	    sceneRadiance = np.zeros(img.shape)
	    for i in range(0, 3):
	        img = np.float64(img)
	        sceneRadiance[:, :, i] = (img[:, :, i] - atomsphericLight) / transmission + atomsphericLight

	    sceneRadiance = np.clip(sceneRadiance, 0, 255)
	    sceneRadiance = np.uint8(sceneRadiance)

	    return transmission,sceneRadiance

	np.seterr(over='ignore')
	if __name__ == '__main__':
	    pass

	path = folder + "/Input"
	files = os.listdir(path)
	files =  natsort.natsorted(files)

	for i in range(len(files)):
	    file = files[i]
	    filepath = path + "/" + file
	    prefix = file.split('.')[0]
	    if os.path.isfile(filepath):
	        print('********    file   ********',file)
	        img = cv2.imread(folder +'/Input/' + file)
	        transmission, sceneRadiance = getRecoverScene(img)
	        cv2.imwrite(folder + '/Input/' + 'DCP_TM.jpg', np.uint8(transmission * 255))
	        cv2.imwrite(folder + '/Input/' + 'DCP.jpg', sceneRadiance)

def restoreMIP(folder):
	np.seterr(over='ignore')
	if __name__ == '__main__':
	    pass

	path = folder + "/Input"
	files = os.listdir(path)
	files =  natsort.natsorted(files)

	for i in range(len(files)):
	    file = files[i]
	    filepath = path + "/" + file
	    prefix = file.split('.')[0]
	    if os.path.isfile(filepath):
	        print('********    file   ********',file)
	        img = cv2.imread(folder +'/Input/' + file)

	        blockSize = 9

	        largestDiff = DepthMap(img, blockSize)
	        transmission = getTransmission(largestDiff)
	        transmission = Refinedtransmission(transmission,img)
	        AtomsphericLight = getAtomsphericLight(transmission, img)
	        sceneRadiance = sceneRadianceRGBMIP(img, transmission, AtomsphericLight)

	        cv2.imwrite(folder + '/Input/' + 'MIP_TM.jpg', np.uint8(transmission * 255))
	        cv2.imwrite(folder + '/Input/' + 'MIP.jpg', sceneRadiance)