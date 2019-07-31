# SPACEWHALE
## Aerial-trained deep learning networks for surveying cetaceans from satellite imagery
### In review - PLoS ONE
[![DOI](https://zenodo.org/badge/192443561.svg)](https://zenodo.org/badge/latestdoi/192443561)


This repo houses the code and resources for this paper:

Borowicz A, Le H, Humphries G, Nehls G, Höschle C, Kosarev V, Lynch H. Aerial-trained deep learning networks for surveying cetaceans from satellite imagery. *In Review.* 

SPACEWHALE is a workflow for using high-resolution satellite imagery and computer vision techniques to locate whales. It's a collaboration between a team at Stony Brook University (@aborowicz, @lmhieu612, and hlynch from @lynch-lab) and a team from BioConsult and HiDef Aerial Surveying (@blackbawks, G. Nehls, C. Höschle, V. Kosarev). It employs pytorch as a framework to train models how to identify whales in imagery. They train on aerial imagery and then can be used on very high-resolution satellite imagery. We used WorldView-3 and -4 imagery (31cm/px) but other sensors could be used. We provide proprietary aerial imagery (of minke whales) from HiDef down-sampled to 31cm/px and other resolutions could be made available. Similarly, aerial imagery from other providers could be used in place of what is here. 
The paper outlines a method for identifying whales in satellite imagery. It is primarily a proof-of-concept and the code contained here is the release associated with the paper. Further updates can be found at github.com/aborowicz/spacewhale.

We use aerial imagery of whales (and water) downsampled to satellite resolution to train several different convolutional neural nets: ResNet-18, -34, -152, and DenseNet. 
We found whales in satellite imagery using google earth and acquired imagery of these regions and dates from the Digital Globe Foundation. Then we validated our trained model with these images.

In the interest of full transparency in science, we also include all elements of the manuscript including reviewer comments and replies.

## Getting Started

SPACEWHALE runs on the command line. Ideally you should be set up with a GPU - otherwise training is a very long process.
On a GPU, you can expect training to take 1-7 hours with our training set, depending on the model architecture you select.


```31cmAerialImagery.zip``` contains the aerial imagery - this imagery has been downsampled to 31cm/pixel to match high-resolution satellite imagery

'''gen_training_patches.py''' takes in images and chops them into 32px x 32px tiles. It takes as arguments the directory of images to tile ```--root```, the step (how many pixels before starting a new tile) ```--step```, the square tile size in pixels ```---size```, and the output directory ```--output```. For example 
```python gen_training_patches.py --root './water_training' --step 16 --size 32 --output './water_tiles'``` 

```m_util2.py``` houses functions etc. that are called by other scripts

```training_tester_weighted.py``` trains a model using a set of aerial images that you define. Example:
``` python training_tester_weighted.py --name model_1 --data_dir './the_data' --verbose True --epochs 24```
```name``` is what you want to call the model you're about to train
```data_dir``` is the directory with your training data in it. In this case your training data need to be in a dir called *train* and you should point to the dir above it. Inside *train* you need a dir with each of your classes (e.g. *whale* and *water*)
```verbose``` asks whether you want info printed out in the terminal
```epochs``` asks for how many epochs you'd like the model to train

```test_script.py``` validates the model with a test set that you define and kicks out some output such as the precision and recall at each epoch. It also writes out 3 CSVs with the filename, label, and prediction for each image in a separate CSV. Example:
```python test_script.py --data_dir './test_dir' --model model_1 --epochs 24```
```data_dir``` should include two dirs labeled with your classes (exactly as they were for training, e.g. *water* and *whale* in our case). 
```model``` is the trained model that you'll use to test with
```epochs``` refers to the # of epochs in your model

The ```shell_scripts``` dir houses scripts used to send training and validation jobs to the SeaWulf cluster at IACS at Stony Brook U (with proper credentials) for Slurm and Torq
The ```Revision_PLOS``` dir houses the working draft of the revised manuscript for this project.

```SPACEWHALE_confusionMatrix.R``` is an R script for building a confusion matrix in ggplot2.

The ```Revision_PLOS``` dir houses the working draft of the revised manuscript for this project.



## Pre-trained Models Used for Training:

```DenseNet161```,  ```ResNet18```, ```ResNet34```, and ```ResNet152``` can be downloaded via torchvision ModelZoo at :

https://pytorch.org/docs/stable/torchvision/models.html

A ResNeXt101 model pre-trained on ImageNet can be downdloaded at:

https://drive.google.com/open?id=1EDUcaGNiakWO9Xvk9kWgkkcnTYZ6VQoT

Please download this pre-trained ResNeXt101 model to the ```RESNEXT``` folder.

## Step-wise

1. Start by tiling your images using the ```gen_training_patches.py``` script. Different models require different input sizes. We tiled everything to 32x32 px.
1. To make other models work (DenseNet requires 224x224px input), we use ```resize_images.py``` to stretch these tiles 
1. Now we can train our model on the aerial imagery using ```training_tester_weighted.py``` We need to choose a model type here. ResNet-18 was our best, with a learning rate set at 0.0009 (set this in the training script).
1. Once the model is trained, it's saved in a dir called "trained_models", we'll call that in the testing step. 
1. You can do a validation step here - we did a 10-fold validation, where we separate the aerial set into 10 parts and train the model on 9, test on the last, then change which 9. We're trying to see if any part of the dataset is controlling the model more than it should.
1. When you're ready for primetime, we can test the model on satellite imagery! You'll need to get your own from an imagery provider - we used Digital Globe and got an imagery grant from the DG foundation. You'll need it to be pretty high resolution.
1. Use the ```test_script.py``` script to find some whales! Use the last training epoch. It outputs 3 .csvs that contain the image filename, the label (the true state) and the prediction so you can see which it missed.
