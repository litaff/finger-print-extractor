import cv2
import numpy as np

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 16
    ksize = 20  # The local area to evaluate
    sigma = 2.0  # Larger Values produce more edges
    lambd = 20.0
    gamma = 0
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
# This general function is designed to apply filters to our image
     
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

def extract_fingerprint(image_path):
    gfilters = create_gaborfilter()
    
    # 0. Load the image
    image = cv2.imread(image_path)

    # 1. Change contrast and brightness to 0.8 and 25
    contrast = 0.8
    brightness = 25
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    # 2. Convert from RGB to Gray
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    # 3. Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    # 4. Normalize image
    normalized = cv2.normalize(equalized, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX)

    # 5. Adaptive thresholding (gaussian c)
    adaptThresharg1 = 67
    adaptThresharg2 = 2
    thresh = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptThresharg1, adaptThresharg2) 

    # 6. Smoothing
    blurrArg = 5
    blurred = cv2.GaussianBlur(thresh, (5, 5), blurrArg)

    # 7. Change contrast and brightness to 1.7 and -40
    contrast2 = 1.7
    brightness2 = -40
    adjusted2 = cv2.convertScaleAbs(blurred, alpha=contrast2, beta=brightness2)

    # 8. Smoothing
    blurr2Arg = 5
    blurred2 = cv2.GaussianBlur(thresh, (5, 5), blurr2Arg)

    # Apply gabor filter
    gabor = apply_filter(blurred2, gfilters)

    # Equalize
    equalized2 = cv2.equalizeHist(gabor)
    equalized3 = cv2.equalizeHist(gabor)

    # Adaptive thresholding (gaussian c)
    adaptThresh2arg1 = 67
    adaptThresh2arg2 = 2
    thresh2 = cv2.adaptiveThreshold(equalized3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptThresh2arg1, adaptThresh2arg2)

    # try extract finger based on contour, works but if the contour is not continous it breaks and shows the original

    # Apply a binary threshold to create a black and white image
    _, binary_image = cv2.threshold(thresh2, 128, 255, cv2.THRESH_BINARY)

    # Create a closing kernel
    closing_kernel_size = 1 # keep at low values
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)

    # Perform morphological closing to bridge small gaps in the contour
    closed_image = cv2.morphologyEx(closing_kernel, cv2.MORPH_CLOSE, closing_kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty mask
    mask = np.zeros_like(thresh2)

    # Draw the largest contour on the mask
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Extract the object using the mask
    result = cv2.bitwise_and(thresh2, mask)

    # Assign final variables
    inputImage = gray
    tracedImage = thresh2
    resultImage = blurred2

    # Resize images while maintaining aspect ratio
    height1, width1 = inputImage.shape
    height2, width2 = tracedImage.shape
    height3, width3 = resultImage.shape 

    # Calculate new height for both images
    new_width = 400
    new_height = int(new_width * max(height1/width1, height2/width2))

    inputImage = cv2.resize(inputImage, (new_width, new_height))
    tracedImage = cv2.resize(tracedImage, (new_width, new_height))
    resultImage = cv2.resize(resultImage, (new_width, new_height))

    # Group into horizontal layout
    horizontal = np.concatenate((inputImage, tracedImage, resultImage), axis=1)

    # Display the result
    cv2.imshow(image_path, horizontal)
    
    

#extract_fingerprint('finger_werav1.jpg')
#extract_fingerprint('finger_werav1_light_crop.jpg')
#extract_fingerprint('finger_werav2.jpg')
#extract_fingerprint('finger_werav2_light_crop.jpg')
#extract_fingerprint('finger_werav3.jpg')
#extract_fingerprint('finger_werav3_light_crop.jpg')
#extract_fingerprint('finger_werav4.jpg')
#extract_fingerprint('finger_werav4_light_crop.jpg')
extract_fingerprint('finger.jpg')
#extract_fingerprint('finger_cropped.jpg')
extract_fingerprint('finger_piotr.jpg')
#extract_fingerprint('finger_piotr_cropped.jpg')
#extract_fingerprint('fingerv2.jpg')
#extract_fingerprint('fingerv2_cropped.jpg')
#extract_fingerprint('finger_pawelv1.jpg')
#extract_fingerprint('finger_pawelv2.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()


