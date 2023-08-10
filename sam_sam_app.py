

import streamlit  as st
import torch
import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import gc
import sys
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import Dataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel 
from torch.optim import Adam
import monai
from skimage import measure
from tqdm import tqdm
from statistics import mean
import torch
import os
from PIL import Image,ImageDraw
from torch.nn.functional import threshold, normalize
torch.cuda.empty_cache()
gc.collect()




def image_processing_qr(x):
       
       
        def get_bounding_box(ground_truth_map):
            # get bounding box from mask
            y_indices, x_indices = np.where(ground_truth_map > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = ground_truth_map.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = [x_min, y_min, x_max, y_max]
            return bbox
        def image_file(image_directory):
           
            ss=[]
            sorted_file_names = sorted(os.listdir(image_directory))
            for file_name in sorted_file_names:
                 # Make sure it's a PNG file (optional check)
               if file_name.endswith(".png"):
                     # Construct the full path to the image
                     image_path = os.path.join(image_directory, file_name)

                     # Open the image
                     ss.append(image_path)
            return ss
        def mask_file(mask_directory):
           
            ss=[]
            sorted_file_names = sorted(os.listdir(mask_directory))
            for file_name in sorted_file_names:
                 
               if file_name.endswith(".tiff"):
                     # Construct the full path to the image
                     mask_path = os.path.join(mask_directory, file_name)

                     # Open the image
                     ss.append(mask_path)
            return ss
        class CustomDataset(Dataset):
            def __init__(self, image_list, mask_list):
                self.image_list = image_list
                self.mask_list = mask_list

            def __len__(self):
                return len(self.image_list)

            def __getitem__(self, key):
                
                if isinstance(key, int):  # If an integer index is provided
                    image = self.image_list[key]
                    mask = self.mask_list[key]
                    return {"image": image, "mask": mask}
                elif key == 'image':  # If 'image' key is provided
                    return self.image_list
                elif key == 'mask':  # If 'mask' key is provided
                    return self.mask_list
                else:
                    raise KeyError(f"Key '{key}' not supported. Use integer index, 'image', or 'mask'.")

            def __str__(self):
                return f"Dataset({{ features: ['image', 'mask'], num_rows: {len(self)} }})"
        
        train_file_path_image="/home/mleng/Pictures/dimension_measurement_sam/f_train"
        t1=image_file(train_file_path_image)  
        train_file_path_mask="/home/mleng/Pictures/dimension_measurement_sam/f_mask"
        t1_mask=mask_file(train_file_path_mask)
        dd = CustomDataset(t1, t1_mask)
        device='cuda'
        image_path=dd[x]["image"]
        mask_path=dd[x]['mask']
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        k=np.array(image)
        
        im = Image.fromarray(k)
        
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
        # Convert the images to NumPy arrays
        ground_truth_mask = np.array(mask)
        prompt = get_bounding_box(ground_truth_mask)
        
        #savinging the original image with bounding box
        x_min, y_min, x_max, y_max = prompt

        # Draw bounding box on the image
        draw1 = ImageDraw.Draw(im)
        draw1.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
        output_path_img = os.path.join("/home/mleng/Pictures/dimension_measurement_sam/demo_img/",'img'+str(x)+'.png')  # Replace with your desired file path and extension
        im.save(output_path_img)
        # prepare image + box prompt for the model
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
        #inputs = {k: v.squeeze(0) for k, v in inputs.items()}

          
          
        model = SamModel.from_pretrained("facebook/sam-vit-base")

        # Load the saved model state dictionary
        model_load_path = "/home/mleng/Pictures/dimension_measurement_sam/fine_tuned_sam2.pth" 
        model_state_dict = torch.load(model_load_path)

        # Load the state dictionary into the model
        model.load_state_dict(model_state_dict)
        model.eval()
        model.to(device)
        # forward pass
        with torch.no_grad():
          outputs = model(**inputs,multimask_output=False)
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask(array)
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob >0.5).astype(np.uint8)
        pil_binary_image = Image.fromarray(medsam_seg.astype('uint8') * 255)
        #converting from pil to numpy array
        pil_binary_image = np.array(pil_binary_image)
        #numpy array converted back to PIL image
        pil_binary_image = Image.fromarray(pil_binary_image)
      
        
        pil_binary_image = pil_binary_image.convert("RGBA")
        # Save the PIL image to a file
        x_min, y_min, x_max, y_max = prompt
        # Draw bounding box on the binary image
        draw = ImageDraw.Draw(pil_binary_image)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
        output_path = os.path.join("/home/mleng/Pictures/dimension_measurement_sam/demo_predicted/",'img'+str(x)+'.png')  # Replace with your desired file path and extension
        pil_binary_image.save(output_path)
        xx='img'+str(x)+'.png'
        return xx


def main():
    st.markdown("<h1 style='text-align: left; color: blue;'>QR code Detection </h1>", unsafe_allow_html=True)
    number =st.number_input('Enter a number', min_value=0, max_value=350)
    if number is not 0:
      img_name=image_processing_qr(number)
      col1, col2= st.columns(2)
      if st.button('Process'):
                         
         with col1:
            st.header('Original_image')
            st.image(os.path.join('/home/mleng/Pictures/dimension_measurement_sam/demo_img',img_name))
         with col2:
            st.header('Predicted_mask')
            st.image(os.path.join('/home/mleng/Pictures/dimension_measurement_sam/demo_predicted',img_name))
     
    
if __name__=='__main__':
    main()
    

