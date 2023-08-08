from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import Dataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel 
from torch.optim import Adam
import monai
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from statistics import mean
import torch
import os
from PIL import Image
from PIL import ImageDraw
from skimage import measure
from torch.nn.functional import threshold, normalize

#dataset = load_dataset("nielsr/breast-cancer", split="train")
#print(dataset[0])

def image_file(image_directory):
   
    ss=[]
    nam_img=[]
    sorted_file_names = sorted(os.listdir(image_directory))
    for file_name in sorted_file_names:
         # Make sure it's a PNG file (optional check)
       if file_name.endswith(".png"):
             # Construct the full path to the image
             image_path = os.path.join(image_directory, file_name)
             
             # Open the image
             ss.append(Image.open(image_path))
             #ss.append(image_path)
             nam_img.append(file_name)
    return ss,nam_img
        
train_file_path_image="/home/mleng/Pictures/dimension_measurement_sam/f_train"
t1,nam_train=image_file(train_file_path_image)       
tt_file_path_image="/home/mleng/Pictures/dimension_measurement_sam/f_test"
tt1,nam_test=image_file(tt_file_path_image) 

def mask_file(mask_directory):
   
    ss=[]
    nam_img_m=[]
    sorted_file_names = sorted(os.listdir(mask_directory))
    for file_name in sorted_file_names:
         
       if file_name.endswith(".tiff"):
             # Construct the full path to the image
             mask_path = os.path.join(mask_directory, file_name)

             # Open the image
             ss.append(Image.open(mask_path))
             #ss.append(mask_path)
             nam_img_m.append(file_name)
    return ss,nam_img_m
train_file_path_mask="/home/mleng/Pictures/dimension_measurement_sam/f_mask"
t1_mask,nam_mask_train=mask_file(train_file_path_mask)       
tt_file_path_mask="/home/mleng/Pictures/dimension_measurement_sam/f_mask_t"
tt1_mask,nam_mask_test=mask_file(tt_file_path_mask)
#print(t1[1])
#print('*'*100)
#print(t1_mask[1]) 

writer = SummaryWriter("/home/mleng/Pictures/expirement1/tensorboard/logs")
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

dataset_train = CustomDataset(t1, t1_mask)
dataset_val = CustomDataset(tt1, tt1_mask)


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



class SAMDataset(Dataset):
    def __init__(self, dataset, processor, image_names, mask_names):
        self.dataset = dataset
        self.processor = processor
        self.image_names = image_names
        self.mask_names = mask_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["mask"])
        image_name = self.image_names[idx]  # Get the corresponding image name
        mask_name = self.mask_names[idx]  # Get the corresponding mask name

        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask
        inputs["image_name"] = image_name
        inputs["mask_name"] = mask_name

        return inputs



processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset_train, processor=processor,image_names=nam_train, mask_names=nam_mask_train)
val_dataset= SAMDataset(dataset=dataset_val, processor=processor,image_names=nam_test, mask_names=nam_mask_test)
example = train_dataset[0]




train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
batch = next(iter(train_dataloader))
#for k,v in batch.items():
 # print(k,v.shape)


model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)
    


# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 70

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


for epoch in range(num_epochs):
    epoch_losses_train = []
    model.train()
    c=0
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)

      for idx, (image, prompt, pred_mask,image_name,mask_name) in enumerate(zip(batch["pixel_values"],batch["input_boxes"],predicted_masks,batch["image_name"],batch["mask_name"])):
            # Convert tensors to numpy arrays
            # Convert tensor to original color image format
            image_np = image.permute(1, 2, 0).cpu().detach().numpy()
            image_np = (image_np * 255).astype(np.uint8)  # Con
            # Convert modified numpy array back to a PIL image
            image_pil = Image.fromarray(image_np)
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = prompt[0]

            # Draw bounding box on the image
            draw = ImageDraw.Draw(image_pil)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

            pred_mask_np = pred_mask.squeeze().detach().cpu().numpy()
            medsam_seg = (pred_mask_np > 0.5).astype(np.uint8)  # Threshold the mask array

            # Convert the binary mask array to a PIL image
            pil_binary_image = Image.fromarray(medsam_seg * 255, mode="L")
           
            #converting to numpy array
            pil_binary_image = np.array(pil_binary_image)
            # Find contours using skimage.measure.find_contours
            contours = measure.find_contours(pil_binary_image, 0.5)
            #numpy array converted back to PIL image
            pil_binary_image = Image.fromarray(pil_binary_image)
            # Calculate bounding box coordinates from the contours
             
            bounding_boxes = []
            for contour in contours:
               # Convert the contour points to a list of tuples
               

               # Calculate the bounding box coordinates
               x_min, y_min = np.min(contour, axis=0)
               x_max, y_max = np.max(contour, axis=0)
               bounding_boxes.append((x_min, y_min, x_max, y_max))
            # Draw bounding boxes on the image
            draw1 = ImageDraw.Draw(pil_binary_image)
            for bbox in bounding_boxes:
                x_min, y_min, x_max, y_max = bbox
                draw1.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        
            # Save image with bounding box
            bbox_image_path = os.path.join('/home/mleng/Pictures/expirement1/input image', f"epoch_{epoch}_{image_name}")
            image_pil.save(bbox_image_path)

            # Save predicted mask with bounding box
            bbox_mask_path = os.path.join('/home/mleng/Pictures/expirement1/train prediction' , f"epoch_{epoch}_{mask_name}")
            pil_binary_image.save(bbox_mask_path)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses_train.append(loss.item())

    #z=sum(epoch_losses_train) / len(epoch_losses_train)
    print(f'EPOCH: {epoch}')
    mean_train_loss=mean(epoch_losses_train)
    print(f'Trainloss: {mean(epoch_losses_train)}')
    writer.add_scalar("Training Loss", mean_train_loss, epoch)
    epoch_losses_val = []
    model.eval()

    with torch.no_grad():
         for batch in tqdm(val_dataloader):
             # forward pass
             outputs = model(pixel_values=batch["pixel_values"].to(device),
                             input_boxes=batch["input_boxes"].to(device),
                             multimask_output=False)

             # compute validation loss
             predicted_masks = outputs.pred_masks.squeeze(1)
             ground_truth_masks = batch["ground_truth_mask"].float().to(device)
             val_loss_batch = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
             epoch_losses_val.append(val_loss_batch.item())
         mean_val_loss=mean(epoch_losses_val)
         print(f'Validation loss: {mean(epoch_losses_val)}')
         writer.add_scalar("Validation Loss", mean_val_loss, epoch)
writer.flush()
writer.close()
# Save the model after training
model_save_path = "/home/mleng/Pictures/expirement1/fine_tuned_sam2.pth"
torch.save(model.state_dict(), model_save_path)
