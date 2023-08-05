#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 18:35:47 2023

@author: mleng
"""
import os
from PIL import Image
from datasets import load_dataset
dd = load_dataset("nielsr/breast-cancer", split="train")
print(dd['label'])
"""
folder_path="/home/mleng/Pictures/dimension_measurement_sam/New_train"
folder_path_png="/home/mleng/Pictures/dimension_measurement_sam/f_train"
def resize_images_in_folder(folder_path,folder_path_png, target_size=(256, 256)):
    png_image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # Load the JPG image using Pillow
            jpg_image_path = os.path.join(folder_path, filename)
            jpg_image = Image.open(jpg_image_path)

            # Resize the image to the target size (256x256)
            resized_image = jpg_image.resize(target_size, Image.ANTIALIAS)

            # Save the resized image as PNG
            png_image_path = os.path.join(folder_path_png, os.path.splitext(filename)[0] + ".png")
            resized_image.save(png_image_path, format="PNG")
            
            # Append the PNG image to the list
            png_image_list.append(Image.open(png_image_path))

    return png_image_list
"""
# Replace "path_to_folder" with the actual path to the folder containing JPG images
##Train image
#folder_path = "/home/mleng/Pictures/dimension_measurement_sam/f_train"
#image_list = resize_images_in_folder(folder_path,folder_path_png, target_size=(256, 256))
#print(image_list)
##Test image
#folder_path_tt="/home/mleng/Pictures/dimension_measurement_sam/New_test"
#folder_path_png_tt="/home/mleng/Pictures/dimension_measurement_sam/f_test"
#image_list_tt = resize_images_in_folder(folder_path_tt,folder_path_png_tt, target_size=(256, 256))
#print(image_list_tt)



def resize_mask_in_folder(folder_path_mask,folder_path_png, target_size=(256, 256)):


     # List all files in the input folder
     tiff_image_list=[]
     for file in os.listdir(folder_path_mask):
           # Check if the file is a JPG image
        if file.endswith(".jpg"):
           # Open the JPG image
           jpg_image_path = os.path.join(folder_path_mask, file)
           jpg_image = Image.open(jpg_image_path)

           # Resize the image to the target size (256x256)
           resized_image = jpg_image.resize(target_size, Image.ANTIALIAS)
        
        
           # Convert the image to the desired format (mode=I)
           img_tiff = resized_image.convert("I")
        
            # Save the image in the output folder with a .tiff extension
           tiff_output_path = os.path.join(folder_path_png, os.path.splitext(file)[0] + ".tiff")
           img_tiff.save(tiff_output_path)
           # Append the PNG image to the list
           tiff_image_list.append(Image.open(tiff_output_path))

     return tiff_image_list
#train 
"""     
folder_path_mask="/home/mleng/Pictures/dimension_measurement_sam/mask_train"
folder_path_png_mask="/home/mleng/Pictures/dimension_measurement_sam/f_mask"
mask_list = resize_mask_in_folder(folder_path_mask,folder_path_png_mask, target_size=(256, 256))
print(mask_list)
"""
#test
folder_path_mask="/home/mleng/Pictures/dimension_measurement_sam/mask_test"
folder_path_png_mask="/home/mleng/Pictures/dimension_measurement_sam/f_mask_t"
mask_list = resize_mask_in_folder(folder_path_mask,folder_path_png_mask, target_size=(256, 256))
print(mask_list)



