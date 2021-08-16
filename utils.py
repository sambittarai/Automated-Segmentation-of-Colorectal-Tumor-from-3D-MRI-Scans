import cv2
import numpy as np

def create_mask(imagePath, addInterior=True):
    """ 
    ACTION: 
        Generates a black and white mask of the input image
        The white area corresponds to green markings in the
        file including any interior points and the rest is black. 
        Note that this algorithm fails in some situations, 
        check the output visually to make sure it is correct. 
    INPUTS: 
        imagePath: path to image file
        maskPath: path to mask file to be created
        addInterior: True/False, whether the pixels inside the green line should be labeled white or not
    OUTPUT: 
        1 if mask was created, 0 if not
    """

    # Read image (if it exists) and make copy for comparison
    if cv2.haveImageReader(imagePath):
        originalImage = cv2.imread(imagePath)
    else:
        print(f"Failed to read input file at {imagePath}")
        return 0
    image = originalImage.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range and mask everything within range
    lower = np.array([50, 125, 125], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    if addInterior:
        # Add interior points to mask
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        mask = cv2.fillPoly(mask, cnts, (255, 255, 255))
        mask = erosion(mask) # Perform erosion on mask

    # Save the output
    # if not cv2.imwrite(maskPath, mask):
    #     print(f"Failed to write output file at {maskPath}")

    return mask

def erosion(inputMask):
    """
    ACTION: 
        Performs erosion on the input mask, returns the result
    INPUTS: 
        inputMask: binary mask on which to perform erosion on
    OUTPUT: 
        Eroded version of the input mask
    """
    kernel = np.ones((3,3),np.uint8)
    kernel[0,0] = 0
    kernel[0,-1] = 0
    kernel[-1,0] = 0
    kernel[-1,-1] = 0
    return cv2.erode(inputMask,kernel,iterations = 1)