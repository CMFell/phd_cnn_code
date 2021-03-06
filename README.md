# phd_cnn_code

This code can be used to automatically detect animals in aerial images based on the Greater Fish River Canyon (GFRC) dataset. 

It is designed to use with data from GFRC dataset *add link*. All code assumes a base directory of locally saved data that is structured in the same way.
The data is saved in four folders, train_images, valid_images, test_images and weights.
In the image folders there are sub folders called pos and neg, pos contains images that have animals in them and neg contains images without animals, there is also a csv in each folder that has the bounding boxes of the animals.
There are two options for the models using just the images or in addition add metadata from a statistical model. The metadata from the statistical model is available in the GFRC dataset as preds_for_cnn.csv this data is only available for the images in the GFRC dataset so if you are using other data any metadata flags should be set to False.

The code is contained with the src folder. In the src folder there are six files that can be used to run different parts of the code these are described below. The code can be used to train a network on either the GFRC data of new data. Or it can be used to test the results of pretrained networks whose weights are saved in the weights folder of the GFRC dataset.

If you want to train a new network, you have to prepare the data, train the network then test on validation set to determine best performing epoch. There are separate files to run to carry out each of these steps.

prepare_data.py - The GFRC images are too big to fit through the CNN whole. This file will split the images into smaller tiles that can be processed. Lines 11 and 13 need to be edited before running, you need to set the basedir location where the data is saved locally and select train or valid set. It needs to be run on both the train and valid sets. The other settings in the file are set to those used to train on the GFRC data but can be varied. This creates a subset for training and validation that balances tiles with animals and tiles without animals.

train_yolo.py - This will train the network and save out a checkpoint file for each epoch. You need to set the values basedir to the location where the data is saved locally, and name_out which will determine the name of directories and files that are output from the program. Set flag use_meta to False if you don't want to use the additional metadata from the statistical model.

valid_yolo,py - You will run this after the train_yolo, this will test the network on the validation set and give results that will allow you to select the point where the network is best fitted.  Set flag use_meta to the same as the train_yolo file.


If you want to test a trained network there are three different options,  a demonstration of a manual checking gui.

whole_image_results - run on every tile in all images in the validation and test set and get summary results (e.g recall, FPPI) for the whole set. You need to set the base directory where the images are saved locally, the set to run on (valid or test) and the model to use (without metadata, with metadata or other). There are two sets of pretrained weights one based on just the CNN (without metadata) this can be run on any images including new ones outside of those included in the GFRC dataset. The other includes metadata from a statistical model predicting the probability of animals at that location so will only run on images for which that information is available those in the GFRC dataset. This file needs to be run in order but after each section it outputs results, so that if you want to run it a bit at a time there are a series of flags in lines 27 to 39 that can be used to tell it to read in results of earlier setions instead of running all sections at once.  

process_single_image - visualise the detections on a single image of your choice, this also need the base directory and the model to use to be selected, but also the image to run on and the set it is contained within. There are four flags in lines 29 to 35. predict_results calculates and saves the raw results from the CNN. calculate_threshold_results applies filters and NMS to give the final results on the image. These must both be run in order. The final two flags draw_example_results and manual_check are optional. draw_example_results will write out the image with the boxes of the detections drawn on it green = true positive, red = false negative, yellow = false positive. manual_check will start a gui that will show you a window of the detected area with buttons allowing you to classify manually as an animal or not you can then save these to a csv. 

example_check_results_gui - this is an example gui that could be used for manually checking the results. It takes a set of calculated results saved in data/example_results_for_manual_checking of the sort that would be exported from whole_image_results and then shows a detection with buttons to classify as animal or not. There is also a button to save the results of the manual checking. The example results can work for every image in the test set. 

