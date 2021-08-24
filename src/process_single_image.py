import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd

from postprocess.display_output import display_single_image_results, manual_check_single_image
from postprocess.image_process import get_image_file_names, predict_on_images
from postprocess.results_process import remove_on_size, calculate_threshold_results, single_image_results
from postprocess.truth_process import process_truths
from postprocess.visualise_results import draw_res_single
from yolo.yolo_inference import YoloClass


# Select base directory where downloaded image files and weights saved 
basedir = "/data/GFRC_data/"
# type of input valid or train
settype = 'valid'
image_file = 'pos/Z138_Img02888.jpg'
# Select from one of two pretrained models or other for a newly trained one
# If using an other model you will need to set further parameters in lines 64 to 79
model_to_use = ['without_metadata', 'with_metadata', 'other'][0]

# Use the following flags to set what results to calculate
# Each step need to be carried out in order
# Each step saves results and by setting to False earlier saved results can be read in instead of recalculating

# 1. Use the model to detect animals in all images
predict_results = True
# 2. Filter and NMS results and match against truths
calculate_results_for_threshold = True
# 3. Draw results on image
draw_example_results = True
# 4. manually check output detections
manual_check = True


### select model to use
if model_to_use == 'without_metadata':
    binmulti = 'binary'
    name_out = 'rgb_bin'
    save_dir = basedir + 'output/' + name_out + '/'
    saveweightspath = basedir + "weights/rgb_binary_weights.pt"
    channels_in = 3
    gfrc_conf_threshold = 0.2
    meta_end = False
    col_list = None
elif model_to_use == 'with_metadata':
    binmulti = 'binary'
    name_out = 'rgb_meta'
    save_dir = basedir + 'output/' + name_out + '/'
    saveweightspath = basedir + "weights/combined_ci_weights.pt"
    channels_in = 3
    gfrc_conf_threshold = 0.25
    meta_end = True
    col_list = ['lowerci', 'upperci']
    channels_in = channels_in + len(col_list)
else:
    binmulti = 'binary'
    name_out = 'rgb_bin_retrained'
    save_dir = basedir + 'output/' + name_out + '/'
    # epoch number of weights to use
    nn = 0
    saveweightspath = save_dir + 'testing_save_' + str(nn) + '.pt'
    # RGB so 3 channels change if not
    channels_in = 3
    # set chosen confidence threshold
    gfrc_conf_threshold = 0.5
    # add metadata at start or end
    meta_end = False
    # set list of columns to use for metadata
    col_list = None
    if col_list != None:
        channels_in = channels_in + len(col_list)


if binmulti == 'binary':
    nclazz = 1
    title = "GFRC binary confusion matrix"
    labels = [0, 1]
    categories = ["none", "animal"]
else:
    nclazz = 6
    title = "GFRC multiclass confusion matrix"
    labels = [0, 1, 2, 3, 4, 5, 6]
    categories = ["none", "bok", "oryx", "kudu", "zeb", "ost", "unid"]
    # not animal 0 code=none
    # springbok 1 (0+1) code=bok
    # oryx 2 (1+1) code=oryx
    # kudu 3 (2+1) code=kudu
    # zebra 4 (3+1) code=zeb
    # ostrich 5 (4+1) code=ost
    # unidentified 6 (5+1) code=unid


whole_image_dir = basedir + settype + "_images/"
truth_file = whole_image_dir + settype + "_GFRC_bboxes_" + binmulti + ".csv"
output_save_dir = save_dir + settype + "_results/" + Path(image_file).stem + "/"
Path(output_save_dir).mkdir(parents=True, exist_ok=True)

# Get true positions for comparison
gfrc_truth = pd.read_csv(truth_file)
gfrc_truth = process_truths(gfrc_truth, 'gfrc')

# process image files
if predict_results:
    yolo_model = YoloClass(wtpath=saveweightspath, channels=channels_in, nclazz=nclazz, 
                           meta_cols=col_list, meta_end=meta_end, basedir=basedir)
    gfrc_windows = predict_on_images([whole_image_dir + image_file], yolo_model, basedir)
    windows_filename = name_out + '_' + Path(image_file).stem + '_windows.csv'
    gfrc_windows.to_csv(output_save_dir + windows_filename, index=False)
else:
    windows_filename = name_out + '_' + Path(image_file).stem +'_windows.csv'
    gfrc_windows = pd.read_csv(output_save_dir + windows_filename)

if calculate_results_for_threshold:
    gfrc_truth = gfrc_truth[gfrc_truth.file_loc == Path(image_file).name]
    gfrc_windows = remove_on_size(gfrc_windows)
    gfrc_results = single_image_results(gfrc_windows, gfrc_truth, whole_image_dir, gfrc_conf_threshold, 0.25, 0.25, 'mean')
    results_loc = output_save_dir + name_out + '_' + Path(image_file).stem + "_results.csv"
    gfrc_results.to_csv(results_loc, index=False)
else:
    results_loc = output_save_dir + name_out + '_' + Path(image_file).stem + "_results.csv"
    gfrc_results = pd.read_csv(results_loc)

if draw_example_results:
    image_output_loc = output_save_dir + name_out + '_' + Path(image_file).stem + Path(image_file).suffix
    res_list = [str(int(np.sum(gfrc_results.confmat == 'TP'))), 
                str(int(np.sum(gfrc_results.confmat == 'FP'))), 
                str(int(np.sum(gfrc_results.confmat == 'FN'))),
                results_loc, image_output_loc]
    res_image = draw_res_single(gfrc_results, image_file, whole_image_dir, output_save_dir, name_out)
    display_single_image_results(res_image, image_file, res_list)

if manual_check:
    manual_check_single_image(results_loc, whole_image_dir + image_file)

