
import cv2
import sys 
 
if len(sys.argv) > 1: 
    s = sys.argv[1] 
 
source = cv2.VideoCapture(0) 
image_thresh = 'THRESH_BINARY'

win_name = 'Camera Preview' 
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 
 
while cv2.waitKey(1) != 27: #esc
    has_frame, frame = source.read() 
    if not has_frame: 
        break 
    
    key = cv2.waitKey(1)
    if key == ord("1"): 
        image_thresh = 'THRESH_BINARY'
    elif key == ord("2"): 
        image_thresh = 'THRESH_BINARY_INV'
    elif key == ord("3"): 
        image_thresh = 'THRESH_TRUNC'
    elif key == ord("4"): 
        image_thresh = 'THRESH_TOZERO'
    elif key == ord("5"): 
        image_thresh = 'THRESH_TOZERO_INV'
    elif key == ord("6"): 
        image_thresh = 'THRESH_MASK'
    
    if image_thresh == 'THRESH_BINARY': 
        img_read = cv2.imread(win_name, frame, cv2.IMREAD_GRAYSCALE) 
        retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)
    elif image_thresh == 'THRESH_BINARY_INV': 
        img_read = cv2.imread(win_name, frame, cv2.IMREAD_GRAYSCALE) 
        retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY_INV)
    elif image_thresh == 'THRESH_TRUNC': 
        img_read = cv2.imread(win_name, frame, cv2.IMREAD_GRAYSCALE) 
        retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_TRUNC) 
    elif image_thresh == 'THRESH_TOZERO': 
        img_read = cv2.imread(win_name, frame, cv2.IMREAD_GRAYSCALE) 
        retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_TOZERO) 
    elif image_thresh == 'THRESH_TOZERO_INV': 
        img_read = cv2.imread(win_name, frame, cv2.IMREAD_GRAYSCALE) 
        retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_TOZERO_INV) 
    elif image_thresh == 'THRESH_MASK': 
        img_read = cv2.imread(win_name, frame, cv2.IMREAD_GRAYSCALE) 
        retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_MASK) 

    cv2.imshow(win_name, img_thresh) 
 
source.release() 
cv2.destroyWindow(win_name) 
