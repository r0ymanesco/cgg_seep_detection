# CGG Oil Seep Detection Exercise

## Objective
The objective is to produce a deep convolutional neural network (DCNN) model and a evaluation metric for image segmentation. 

## Dataset 
The given dataset contains 760 synthetic aperture radar images. To train the network, the dataset is first split 80:10:10 between training, validation and evaluation, respectively. They are stored in the folders ```train_images_256/```, ```valid_images_256/```, ```eval_images_256/```, with the corresponding masks stored in ```train_masks_256/```, ```valid_masks_256/```, ```eval_masks_256/```. The folders ```all_images_256/``` and ```all_masks_256/``` contains all the images and masks for final prediction. 

## Run
Prerequisits:
* Numpy
* Pytorch 1.40
* OpenCV 

To run the code, simply run the shell script ```./train.sh``` in the terminal with the prerequits satisfied and the datasets in the folders described above. The final predictions are made across the entire dataset, with the output saved as ```.tif``` files in the ```output/``` folder, just as with the masks provided. The trained model is saved in the ```model/``` directory with the name ```unet.pth```.

## Network Architecture
The architecture of choice for this task is the U-Net, which is a fully convolutional DCNN that have been used in biomedical imaging segmentation. The network first progressively downsamples the input image by passing through 5 convolutional layers and max-pooling after each convolution. The learned features at the final downsampling layer is then passed through 4 upsampling convolutional layers, which uses bilinear upsampling after each convolution to progressively enlarge the prediction. To improve accuracy and stability, residual connections are formed between the downsampling layers and the upsampling layers to allow features of different resolutions to be shared between the two segments of the network. Finally, at the output of the 4th upsampling layer, softmax activation is applied to produce an 8 channel prediction with the same height and width as the input image, where each pixel location contains the predicted probability of the level of seep; that is, the network predicts for each pixel location a vector of probabilities, where the probability at index i represents the probability that the level of seep is i. 

## Loss Function
The loss function chosen for this task is the cross entropy loss. The reason this loss function is chosen is because the minimisation of the cross entropy corresponds to the maximum likelihood estimation of the network parameters with respect to the likelihood distribution of the decision given the dataset. 

## Training Details
Training was done on batch size 16, using the Adam optimiser with learning rate 0.001. The maximum epoch was set at 1000, with early stopping to prevent overfitting. The early stopping patience was 10 epochs, with minimum loss improvement 0.001 required. 

