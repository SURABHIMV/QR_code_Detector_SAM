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
from torch.nn.functional import threshold, normalize

#dataset = load_dataset("nielsr/breast-cancer", split="train")
#print(dataset[0])

def image_file(image_directory):
   
    ss=[]
    for file_name in os.listdir(image_directory):
         # Make sure it's a PNG file (optional check)
       if file_name.endswith(".png"):
             # Construct the full path to the image
             image_path = os.path.join(image_directory, file_name)

             # Open the image
             ss.append(Image.open(image_path))
    return ss
        
train_file_path_image="/home/mleng/Pictures/dimension_measurement_sam/f_train"
t1=image_file(train_file_path_image)       
tt_file_path_image="/home/mleng/Pictures/dimension_measurement_sam/f_test"
tt1=image_file(tt_file_path_image) 

def mask_file(mask_directory):
   
    ss=[]
    for file_name in os.listdir(mask_directory):
         
       if file_name.endswith(".tiff"):
             # Construct the full path to the image
             mask_path = os.path.join(mask_directory, file_name)

             # Open the image
             ss.append(Image.open(mask_path))
    return ss
train_file_path_mask="/home/mleng/Pictures/dimension_measurement_sam/f_mask"
t1_mask=mask_file(train_file_path_mask)       
tt_file_path_mask="/home/mleng/Pictures/dimension_measurement_sam/f_mask_t"
tt1_mask=mask_file(tt_file_path_mask) 

writer = SummaryWriter("/home/mleng/Pictures/dimension_measurement_sam/tensorboard/logs")
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
dataset_val = CustomDataset(t1, t1_mask)


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
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["mask"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs 



processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
train_dataset = SAMDataset(dataset=dataset_train, processor=processor)
val_dataset= SAMDataset(dataset=dataset_val, processor=processor)
example = train_dataset[0]
for k,v in example.items():
  print(k,v.shape)



train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)


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
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
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
model_save_path = "/home/mleng/Pictures/dimension_measurement_sam/fine_tuned_sam2.pth"
torch.save(model.state_dict(), model_save_path)

