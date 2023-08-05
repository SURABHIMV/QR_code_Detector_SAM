import streamlit  as st
import torch
import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import gc
import sys
from transformers import SamProcessor
from PIL import Image
import pandas as pd
import os
import json
import math
from transformers import SamModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
#To empty the cache
torch.cuda.empty_cache()
gc.collect()



# path to original image in static folder(dm)
image_path_dm= "/home/mleng/Pictures/dimension_measurement_sam/statics/"
# # path to contour image in upload folder(dm)
output_path_dm= "/home/mleng/Pictures/dimension_measurement_sam/upload/"  

image_path_qr= "/home/mleng/Pictures/dimension_measurement_sam/train/"

output_path_qr= "/home/mleng/Pictures/dimension_measurement_sam/upload_br/"  

def image_processing_dm(x):
    #sys.path.append("..")
    image = cv2.imread(x)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    sam_checkpoint = "/home/mleng/Pictures/dimension_measurement_sam/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = "cuda"

    #model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    #mask generator

    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100  # Requires open-cv to run post-processing
    )

    masks = mask_generator_.generate(image)
    # Loop through each mask in the masks variable and draw contours
    c=0
    H=[]
    W=[]
    for ann in masks:
        
        m = ann['segmentation']
        if m.sum() > 0:
            # Convert the binary mask to an 8-bit unsigned integer image
            mask_uint8 = (m * 255).astype(np.uint8)
            
            # Find contours in the binary mask image
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_list = [contour.tolist() for contour in contours]
            num_contours=len(contours_list)
            
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            c=c+1
            
            contour_img = np.zeros_like(image)
            cv2.drawContours(contour_img, [contours[0]], -1, (255, 255, 255), thickness=2)
                
            # Get bounding box coordinates and draw bounding box on contour_img
            x, y, w, h = cv2.boundingRect(contours[0])
            H.append(h)
            W.append(w)
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                
            # Save the modified contour image with a unique filename
            #fname ='/home/mleng/Pictures/dimension_measurement_sam/upload'
            #os.makedirs(fname, exist_ok=True)  # Create directory if it doesn't exist
            image_path = os.path.join('/home/mleng/Pictures/dimension_measurement_sam/upload', f'{c}_contour' + '.png')
            cv2.imwrite(image_path, contour_img)
            
    return H,W,c   
    


def extract_contour_number(filename):
    return int(filename.split("_")[0])

def main():
    st.markdown("<h1 style='text-align: left; color: blue;'>Dimension measurement </h1>", unsafe_allow_html=True)
    # menu = ['Image Based', 'Video Based']
    menu = ['Dimension measurement']
    st.sidebar.header('Mode Selection')
    choice = st.sidebar.selectbox('What is your data source ?', menu)
    if choice=='Dimension measurement':
        
        #image_files=[file for file in os.listdir(output_path_dm) if file.lower().endswith(('.png'))]
        #sorted_pr = sorted(image_files)
        loaded_file = st.file_uploader("Choose a image",type=["jpg","png"])
        
        if loaded_file is not None:
            z = image_path_dm + loaded_file.name
            for file in os.listdir(output_path_dm):
                file_path = os.path.join(output_path_dm, file)
                os.remove(file_path)
        
            hh, ww, nc = image_processing_dm(z)
            col1, col2, col3, col4, col5 = st.columns(5)
            
            if st.button('Process'):
                
                    sorted_pr =sorted(os.listdir(output_path_dm), key=extract_contour_number)
                    data={'Sr.no':list(range(len(hh))),'Contour_image':[output_path_dm + i for i in sorted_pr],'pixel_l':hh,'pixel_w':ww}
                    df=pd.DataFrame(data)
                    df1=df[['Sr.no','Contour_image','pixel_l','pixel_w']]
                    # Display the DataFrame
                    st.dataframe(df1)

                    # Display images in the 'Contour_image' column
                    st.header("Contour images")
                    for i in range(len(sorted_pr)):
            
                       st.subheader(f"{i+1}_Contour")
                       st.image(df1['Contour_image'][i], width=300)
    
     
    
if __name__=='__main__':
    main()
    