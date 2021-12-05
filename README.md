# Generating High Resolution Image from Low Resolution Image Using Hybrid Neural Network of DBSRCNN, DNSRCNN, and ESPCNN

## This deep learning project was done for a class.

# Abstract

In this modern era, having high resolution-colored images has never been easier. There are numerous tools that enable people to get high resolution-colored images, however, there are several circumstances where having such kind of images is extremely difficult or hard to get. The aim is to reduce the problems and hassle of identifying criminals caught by police dash cams and CCTV cameras. We also want to focus on generating high resolution image from low resolution image for people using phones with lower resolution cameras. So, in the hopes of improving that aspect, we plan to turn low-resolution grey images to high-resolution color images. By using our algorithm, phone companies can use inexpensive cameras with low resolution and the image taken with those cameras will go through our algorithm and turn them to high quality and sharp images and modify those images taken by police dash cams and CCTV cameras.

# Implementation details

GPU == Nvidia Quadro RTX 8000
Keras == 2.4.3
Tensorflow == 2.4.1
Tensorboard == 2.4.0
SciPy == 1.7.1
Python == 3.9.7
MatplotLib == 3.4.2
Numpy == 1.21.3
Opencv-python == 4.5.3.56

A requirements.txt file can also be found. 

# Starting Codes and our implementation version

Our base code for all the models were of Efficient Sub-pixel CNN from Keras’ website. We changed the process_input method from ‘area’ to ‘bicubic’. Changed variable names to make it easy for us to understand. We also mentioned the whole dataset path instead of using ‘os.path.join’ command used in the dataset. We used changed our metric to accuracy too. We used our own dataset path instead of using Berkeley dataset as mentioned on the website of the code. Testing has been done on all models for 10, 50, 100, 1000, 4000 images but the default is 10.

## Starting codes links

Dataset: https://github.com/NVlabs/ffhq-dataset
Deep Denoising (DDSRCNN) and Denoising (Autoencoder): https://github.com/titu1994/ Image-Super-Resolution
Deblurring CNN: https://github.com/Fatma-ALbluwi/ DBSRCNN/blob/master/DBSRCNN_train/DBSRCNN_train.py
Efficient Sub-Pixel CNN: https://keras.io/examples/vision/super_resolution_sub_pixel/


## Difference between the starting codes and our own implementation codes

ESPCNN: Keras’ website had different convolutional layers which were not working great for us so we changed the filter sizes for 3 of the 4 convolutional layers as well. For ESPCNN we also got rid of the zooming inside the displayed image. We changed the learning rate from 0.001 to 0.0001. They also ran the images for 100 Epochs, and we ran it for 50 Epochs.

DSRCNN: Similarly, with ESPCNN we changed the filter sizes of the convolutional layer as well as the Epochs to 50. We added the last layer of ‘tf.nn.depth_to_space’ which was not in the original model. Learning rate again was changed from 0.001 to 0.0001. 

DDSRCNN: Similar to the previous models filter sizes of the convolutional layers were changed as well as Epochs to 50. We added the last layer of ‘tf.nn.depth_to_space’ which was not in the original model. Learning rate again was changed from 0.001 to 0.0001.

DBSRCNN: Last convolution layer was changed as well as added the last layer of ‘tf.nn.depth_to_space’ which was not in the original model. Learning rate again was changed from 0.001 to 0.0001.

Our Model: Used the base Keras’ code like the previous models only the changed the network to make a hybrid model. Also, the learning rate was changed from 0.001 to 0.0001 and one of the utility methods had zooming inside the displayed image.  

# Datasets

## Dataset 

We found the dataset in https://github.com/NVlabs/ffhq-dataset which directed us to the link below where we obtained the data. 

Dataset (1024 x 1024) link: https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

In the file data_preprocess.ipynb, we resized the images from 1024 x 1024 to 256 x 256. Then in the code files ESPCNN.ipynb, DSRCNN.ipynb, DDSRCNN.ipynb, DBSRCNN.ipynb, and our_model.ipynb we preprocess the data again to process the target and process the input. 


## Preprocessing 

If there is any pre-processing for your dataset, please specify the file you use for pre-processing. The following is just an example.

The code for preprocessing the data is in the model codes. The preprocessing happens in each of the jupyter notebooks. Image resizing was done in data_preprocess.ipynb but we are uploading the resized images since original images were about 2.64 TB. The file data_preprocess is not needed if using our uploaded data. 

# Execution 

I run the code in the following steps:

(1). I specify the dataset paths in the files ESPCNN.ipynb, DSRCNN.ipynb, DDSRCNN.ipynb, DBSRCNN.ipynb, and our_model.ipynb. All the files will have dataset paths in the first cell. The training data path will be called ‘train_high_ds’ and validation data path will be called ‘validate_high_ds’. The test data path will be called test_path.  


(2). I train the model by running the third cell of the ipynb files. 

The ESPCNN model is trained for about 15 minutes.
The DSRCNN model is trained for about 21 minutes.
The DDSRCNN model is trained for about 33 minutes.
The DBSRCNN model is trained for about 13 minutes
Our Model (our_model.ipynb) is trained for about 35 minutes.


(3). I test the model by running the third cell of the ipynb files. The maximum time taken (4000 images) to test is given below, but time taken will depend on hardware to hardware.

The ESPCNN model is tested for about 34 minutes.
The DSRCNN model is tested for about 48 minutes.
The DDSRCNN model is tested for about 1 hour and 3 minutes.
The DBSRCNN model is tested for about 43 minutes
Our Model (our_model.ipynb) is tested for about 1 hour and 7 minutes.


