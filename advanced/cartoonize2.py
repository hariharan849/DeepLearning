import cv2

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret, originalmage = cap.read()
    #imout = cv2.edgePreservingFilter(originalmage, flags=cv2.RECURS_FILTER)
    #imout = cv2.edgePreservingFilter(originalmage, flags=cv2.NORMCONV_FILTER);
    # Detail enhance filter
    #imout = cv2.detailEnhance(originalmage)

    # Pencil sketch filter
    imout_gray, imout = cv2.pencilSketch(originalmage, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    # Stylization filter
    #cv2.stylization(originalmage,imout)
    cv2.imshow("test", imout)
    cv2.imshow("test", imout_gray)
    c = cv2.waitKey(delay=30) 
    if c == 27: 
        break 

cap.release()
cv2.destroyAllWindows() 
    