# QR_code_Detector_SAM

This repository contains code to detect QR code in an image Using finetuned SAM model. The web application used is Streamlit.

## Roadmap to reproduce the result
* In Anaconda create a new environment within this environment install all the required libraries as mentioned in (Libraries and Versions) and within this same environment open any IDE(spyder, Pycharm, etc).
* Install the hugging face transformer(as given in the libraries and versions) using this transformer SAM model and Sam Processors are used and also install the Monai repository as we'll use a custom loss function from it. As all the installation is done then if the (original images and the mask images are in 'jpg' format) then these images are formatted such that the original image which is in '.jpg' format is converted into '.png' format with shapes (256,256) and the mask image to '.tiff' and this mask and original images are stored in separate folders (all this formatting part of code is in Image_formatting.py file). Using the original image and mask image a dataset is formed (for train and validation separately) using the 'class CustomDataset(Dataset)' which is in the sam_sam.py file. This dataset is passed to SamProcessor (it will do all the required processing of images and their shapes which is needed by the SAM model).then this processed data is passed to the data loader which will provide batches of data to the model for that we have to set the batch size. Then based on the number of epochs set the model is trained with this batch of train data and it is tested using validation data. Train and test results progress can be visualized using the Tensorboard. Once the model is finetuned then saved the model checkpoints to desired path. This fine-tuning part of the code is in sam_sam.py.
* This fined-tuned checkpoint of the model is used in the original SAM model. This model is used for the deployment of QR code data. This app deployment part of the code is in sam_sam_app.py .
* In the app we can see an input widget where we have to enter the number. Based on the number entered it is taken as the index value in the dataset formed. so based on this index it will extract the image and mask image then passed to Samprocessor and then model and get the result. In order to view the original image and predicted image in app also form two separate folders so that after storing it in that folder we can fetch it and show the result in the app.
 

## Files
* `sam_sam.py`: Contains the code regarding fine-tuning the SAM model.
* `Image_formating`: Contains code regarding formatting the original and mask image as required by the SAM model.
* `sam_sam_app.py`: Contains the code regarding deploying the finetuned SAM model using the QR code dataset on the Streamlit application.


## Libraries and Versions

The following libraries and versions were used in this project:
The code requires python>=3.8, as well as Pytorch >=1.7 and Torchvision >=0.8. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.
* `torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116`
* `Streamlit==1.25.0`
* `numpy==1.24.3`
* `pandas==2.0.3`
* `opencv-python==4.8.0.7`
* `python==3.8.16`
* `spyder==5.4.3`
* `pillow==9.4.0`
* `pip install git+https://github.com/huggingface/transformers.git`
* `pip install monai'
* `pip inatall tensorboard`





















