import cv2
import numpy as np

def filter_color(img, lower_range =(0,128,96), upper_range = (180, 255, 255)):#h should be 0-180
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# lower range of red color in HSV
	#lower_range =(0,128,96) #(7, 0, 0) for yellow pink# #(0,128,96) for background removal#
	# upper range of red color in HSV
	#upper_range = (180, 255, 255 )#(90, 255, 255) for yellow pink# #(180, 255, 255 ) for background removal#
	mask = cv2.inRange(hsv_img, lower_range, upper_range)
	color_image = cv2.bitwise_and(img, img, mask=mask)
	# Display the color of the image
	cv2.imwrite('filtered.png', color_image)
	cv2.imshow('filtered', color_image)
	cv2.waitKey(0)
	return color_image, mask
	
def find_connectivity(img):
	j=0
	_, mask = filter_color(img)
	output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
	(numLabels, labels, stats, centroids) = output
	# loop over the number of unique connected component labels
	boxes = []
	for i in range(0, numLabels):
	# if this is the first component then we examine the
	# *background* (typically we would just ignore this
	# component in our loop)
		if i == 0:
			text = "examining component {}/{} (background)".format(i + 1, numLabels)
	# otherwise, we are examining an actual connected component
		else:
			text = "examining component {}/{}".format( i + 1, numLabels)
	# print a status message update for the current connected
	# component
		print("[INFO] {}".format(text))
	# extract the connected component statistics and centroid for
	# the current label
		x = stats[i, cv2.CC_STAT_LEFT]
		y = stats[i, cv2.CC_STAT_TOP]
		w = stats[i, cv2.CC_STAT_WIDTH]
		h = stats[i, cv2.CC_STAT_HEIGHT]
		area = stats[i, cv2.CC_STAT_AREA]
		if area > 100 and area < 1000:
			(cX, cY) = centroids[i]
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
			boxes.append([x,y,w,h])
			j = j + 1
			#txt = "{} tubes".format(j)
			print('..............')
			#cv2.circle(img, (int(cX), int(cY)), 4, (255, 255, 255), -1)
	#cv2.putText(img, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   #fontScale=1, color=(255,0,0), thickness=2)
	#cv2.imwrite('connect.png', img)
	cv2.imshow("Output", img)
	cv2.waitKey(0)
	return boxes
	
def determin_color(patch):
	patch = patch.flatten()
	result_yellow = patch[(patch>7)*(patch<90)] #yellow 7-90, pink 0-6 and 170-180
	result_pink1 = patch[(patch>0)*(patch<6)] 
	result_pink2 = patch[(patch>170)*(patch<180)]
	print('yellow number', result_yellow.shape[0])
	print('pink number', result_pink1.shape[0] + result_pink2.shape[0])
	print('total number', patch.shape)
	yellow_p = result_yellow.shape[0] / patch.shape[0]
	pink_p = (result_pink1.shape[0] + result_pink2.shape[0]) / patch.shape[0]
	print('yellow percentage', yellow_p)
	print('pink percentage', pink_p)
	return yellow_p, pink_p


img = cv2.imread('openlight/51.png')

boxes = find_connectivity(img)

mask2 = np.zeros_like(img)
#mask2 = mask2[:,:,0]
for box in boxes:
	x,y,w,h = box[0],box[1],box[2],box[3]
	mask2[y:y+h,x:x+w,:] = 1	
cv2.imshow("Output", img*mask2)
cv2.imwrite('extract_tubes.png', img*mask2)
cv2.waitKey(0)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

p=0
q=0
for box in boxes:
	x,y,w,h = box[0],box[1],box[2],box[3]
	patch = hsv_img[y:y+h,x:x+w,0]
	yellow_p, pink_p = determin_color(patch)
	if yellow_p > pink_p and yellow_p>0.3: #0.2:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
		p = p + 1
	elif yellow_p <= pink_p and pink_p>0.3: #0.2:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
		q = q + 1
	else:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
txt = '{} tubes. {} postive. {} negative.'.format(p+q, p, q)
cv2.putText(img, txt, (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   fontScale=1, color=(255,255,0), thickness=2)

cv2.imwrite('final.png', img)
cv2.imshow('result', img)
cv2.waitKey(0)




cv2.destroyAllWindows()
