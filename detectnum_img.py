from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import pytesseract

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# construct a blob and pass it through the model
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# get rows and columns from scores
# initialize bounding boxes and confidence scores lists
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over rows
for y in range(0, numRows):

	# extract scores and bounding box coordinates
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over columns
	for x in range(0, numCols):
		#determine if probablilty threshold for text detection met
		if scoresData[x] < 0.35:
			continue

		# compute the offset factor
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle and compute cosine and sine for prediciton
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# get width and height of bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# get x and y coordinates for bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# append bounding box and score to list
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over bounding boxes
for (startX, startY, endX, endY) in boxes:

	# scale the bounding box coordinates according to their ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	dX = int((endX - startX) * 0)
	dY = int((endY - startY) * 0)

	(origH, origW) = image.shape[:2]

	# get region of interest
	roi = orig[startY:endY, startX:endX]

	# put region of interest into pytesseract to predict digits
	result = pytesseract.image_to_string(roi, config='--psm 7 --oem 1 -c tessedit_char_whitelist=-:0123456789')
	print(result)

	# draw bounding box around text in the original image
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# display image
cv2.imshow("Text Detection", cv2.resize(orig, (400, 475)))
cv2.waitKey(0)







