import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)	# converts to grayscale
	kernel = 5
	blur = cv2.GaussianBlur(gray, (kernel,kernel), 0)				# smoothes image noise
	canny = cv2.Canny(blur, 50, 150)					# edge detection method
	return canny

def region_of_interest(image):
	height = image.shape[0] # loads the height appointed by the shape
	polygons = np.array([ 	# points of the triangle
		[(200,height), 		# left bottom
		(1100, height), 	# right bottom
		(550, 250)],		# top center
		], 
		np.int32
		)
	mask = np.zeros_like(image) # creates an array of zeros with the same shape of the image
	cv2.fillPoly(mask, polygons, 255) # fills part of the black mask with the area of triangle
	masked_image = cv2.bitwise_and(image, mask) # crops the image by the area of triangle
	return masked_image

def display_lines(image, lines):
	line_image = np.zeros_like(image) # creates an array of zeros with the same shape of the image
	if lines is not None:
		for line in lines:
			# each line is an array with 4 columns
			# ideally, should reshape line to 1 dimension
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, 
				(x1,y1), 		# first point of line
				(x2,y2), 		# second point of line
				(255, 0, 0), 	# color choice: blue
				10				# line thickness
			)
	return line_image

def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	# image.shape (y, x, RBG)
	y1 = image.shape[0]		# base line plot
	y2 = int(y1 * (3/5))	# limits length to 3/5ths of the image height
	x1 = int((y1 - intercept)/slope)	# y = mx + b
	x2 = int((y2 - intercept)/slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	left_fit = []		# displays coords of line on the left side
	right_fit = []		# displays coords of line on the right side
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2), (y1,y2), 1)	# returns a vector of coefficients which describe the slope and intercept
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))
	left_fit_average = np.average(left_fit, axis=0)		# averages lines on the left side
	right_fit_average = np.average(right_fit, axis=0)	# averages lines on the left side
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])


# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 
# 2, 					# precision of 2 pixels  
# np.pi/180, 			# precision of 1 radian
# 100, 				# threshold of 100 votes for a bin
# np.array([]),		# placeholder
# minLineLength=40,	# minimum line length
# maxLineGap=5		# maximum gap between lines
# ) 
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(
# 	lane_image, 
# 	0.8, 			# weight of 0.8 -> makes lane_image darker  
# 	line_image, 	
# 	1,				# weight of 1.0 -> lines more defined
# 	1				# gamma value scaler
# )
# cv2.imshow('result', combo_image)
# cv2.waitKey(0)
# plt.imshow(canny) # used to show the coordinate system surrounding the image
# plt.show()

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
	_, frame = cap.read()
	canny_image = canny(frame)
	cropped_image = region_of_interest(canny_image)
	lines = cv2.HoughLinesP(cropped_image, 
	2, 					# precision of 2 pixels  
	np.pi/180, 			# precision of 1 radian
	100, 				# threshold of 100 votes for a bin
	np.array([]),		# placeholder
	minLineLength=40,	# minimum line length
	maxLineGap=5		# maximum gap between lines
	) 
	averaged_lines = average_slope_intercept(frame, lines)
	line_image = display_lines(frame, averaged_lines)
	combo_image = cv2.addWeighted(
		frame, 
		0.8, 			# weight of 0.8 -> makes lane_image darker  
		line_image, 	
		1,				# weight of 1.0 -> lines more defined
		1				# gamma value scaler
	)
	cv2.imshow('result', combo_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):	# wait 1ms between frames
		break								# end when 'q' is pressed
cap.release()
cv2.destroyAllWindows()