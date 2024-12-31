import numpy as np
#from google.colab.patches import cv2_imshow
import cv2

BLUE = [255, 0, 0]

img1 = cv2.imread(r'..\logo.png')

#cv2.BORDER_CONSTANT - Adds a constant colored border. The value should be given as next argument.
#cv2.BORDER_REFLECT - Border will be mirror reflection of the border elements, like this : fedcba|abcdefgh|hgfedcb
#cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT - Same as above, but with a slight change, like this : gfedcb|abcdefgh|gfedcba
#cv2.BORDER_REPLICATE - Last element is replicated throughout, like this: aaaaaa|abcdefgh|hhhhhhh
#cv2.BORDER_WRAP - Can’t explain, it will look like this : cdefgh|abcdefgh|abcdefg

replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10, 10, 10, 10, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)
print (img1.shape)
print (constant.shape)
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
    cv2.imshow("image", reflect)
    
cv2.destroyAllWindows()