import cv2

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
	ret, originalmage = cap.read()
	#converting an image to grayscale
	grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
	#applying median blur to smoothen an image
	smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
	getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
	  cv2.ADAPTIVE_THRESH_MEAN_C, 
	  cv2.THRESH_BINARY, 9, 9)
	  
	colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)
	cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
	cv2.imshow("test", cartoonImage)
	c = cv2.waitKey(delay=30) 
	if c == 27: 
		break 

cap.release()
cv2.destroyAllWindows() 
	