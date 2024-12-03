############################################################################################
# The program starts from the folder containing all the images with label 1 (fire + smoke) #
# and divides them in fire images and smoke images. It performs a color analysis and       #
# puts in a folder called "2" all the images with fire, that will have a lot of red and    #
# yellow pixels, while all the remaning ones (smoke images) in a folder "1"                #
############################################################################################

import cv2
import numpy as np
import glob
import shutil


directory = 'dl2425_challenge_dataset\\val\\1\\'

def return_n_white_pixels(img_name):
    
    img = cv2.imread(img_name) # load the image
    result = img.copy() # copy the image
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # we're reading the image as an RGB one
    lower_red = np.array([100,0,0]) # the "lowest" acceptable red
    upper_red = np.array([255,50,20]) # the "highest" acceptable red
    mask_red = cv2.inRange(image, lower_red, upper_red) # masks the image keeping only the red pixels
    lower_yellow = np.array([180,60,0]) # the "lowest" acceptable yellow
    upper_yellow = np.array([255,255,130]) # the "highest" acceptable yellow
    mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow) # masks the image keeping only the yellow pixels
    mask = cv2.bitwise_or(mask_yellow, mask_red) # masks the image keeping only the red and yellow pixels
    # result = cv2.bitwise_and(result, result, mask=mask) # applies the mask to the image
    n_white_pix = np.sum(mask == 255) # sums the number of white pixels in the mask (so either red or yellow pixels)
    #cv2.imshow('mask', mask) # shows the mask
    #cv2.imshow('result', result) # shows the original image with the mask
    #cv2.waitKey()
    return n_white_pix

for filename in glob.iglob(f'{directory}/*'):
    name = filename.split("\\")[3] # get the name of an image in the folder
    if "jpg" in name:
        img_name = directory + name
        n_white_pixels = return_n_white_pixels(img_name)
        if n_white_pixels >= 1000:
            fire_directory = "dl2425_challenge_dataset\\val\\1\\1\\"
            shutil.copyfile(img_name, fire_directory + name) # if we have more than 1000 white pixels in the image, we put it in the fire folder
        else:
            smoke_directory = "dl2425_challenge_dataset\\val\\1\\2\\"
            shutil.copyfile(img_name, smoke_directory + name) # if we have less than 1000 white pixels in the image, we put it in the smoke folder
