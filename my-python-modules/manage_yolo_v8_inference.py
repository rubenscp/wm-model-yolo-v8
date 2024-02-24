"""
Institution: Institute of Computing - University of Campinas (IC/Unicamp)
Project: White Mold 
Description: Implements the YOLOv8 neural network model for step of inference.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 06/02/2024
Version: 1.0
This implementation is based on this notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb?authuser=1
"""


# Basic python and ML Libraries
import os
from datetime import datetime
import shutil
from ultralytics import YOLO
from IPython.display import display, Image

# torchvision libraries
import torch

# these are the helper libraries imported.
import utils

# Importing python modules
from manage_log import *
from tasks import Tasks
from entity.AnnotationsStatistic import AnnotationsStatistic

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
NEW_FILE = True

# ###########################################
# Application Methods
# ###########################################

# ###########################################
# Methods of Level 1
# ###########################################

def main():
    """
    Main method that perform inference of the neural network model.

    All values of the parameters used here are defined in the external file "wm_model_faster_rcnn_parameters.json".
    
    """
 
    # creating Tasks object 
    processing_tasks = Tasks()

    # setting dictionary initial parameters for processing
    full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-yolo-v8'

    # getting application parameters 
    processing_tasks.start_task('Getting application parameters')
    parameters_filename = 'wm_model_yolo_v8_parameters.json'
    parameters = get_parameters(full_path_project, parameters_filename)
    processing_tasks.finish_task('Getting application parameters')

    # setting new values of parameters according of initial parameters
    processing_tasks.start_task('Setting input image folders')
    set_input_image_folders(parameters)
    processing_tasks.finish_task('Setting input image folders')

    # getting last running id
    processing_tasks.start_task('Getting running id')
    running_id = get_running_id(parameters)
    processing_tasks.finish_task('Getting running id')

    # setting output folder results
    processing_tasks.start_task('Setting result folders')
    set_result_folders(parameters)
    processing_tasks.finish_task('Setting result folders')
    
    # creating log file 
    processing_tasks.start_task('Creating log file')
    logging_create_log(
        parameters['inference_results']['log_folder'], 
        parameters['inference_results']['log_filename']
    )
    processing_tasks.finish_task('Creating log file')

    logging_info('White Mold Research')
    logging_info('Inference of the model YOLO v8' + LINE_FEED)

    logging_info(f'')
    logging_info(f'>> Set input image folders')
    logging_info(f'')
    logging_info(f'>> Get running id')
    logging_info(f'running id: {str(running_id)}')   
    logging_info(f'')
    logging_info(f'>> Set result folders')

    # getting device CUDA
    processing_tasks.start_task('Getting device CUDA')
    device = get_device(parameters)
    processing_tasks.finish_task('Getting device CUDA')
    
    # creating new instance of parameters file related to current running
    processing_tasks.start_task('Saving processing parameters')
    save_processing_parameters(parameters_filename, parameters)
    processing_tasks.finish_task('Saving processing parameters')

    # copying weights file produced by training step 
    processing_tasks.start_task('Copying weights file used in inference')
    copy_weights_file(parameters)
    processing_tasks.finish_task('Copying weights file used in inference')

    # creating neural network model 
    processing_tasks.start_task('Creating neural network model')
    model = get_neural_network_model(parameters, device)
    processing_tasks.finish_task('Creating neural network model')

    # getting statistics of input dataset 
    processing_tasks.start_task('Getting statistics of input dataset')
    annotation_statistics = get_input_dataset_statistics(parameters)
    show_input_dataset_statistics(annotation_statistics)
    processing_tasks.finish_task('Getting statistics of input dataset')    

    # inference the neural netowrk model
    processing_tasks.start_task('Running inference of test images dataset')
    inference_neural_network_model(parameters, device, model)
    processing_tasks.finish_task('Running inference of test images dataset')

    # showing input dataset statistics
    show_input_dataset_statistics(annotation_statistics)

    # printing metrics results 
   
    # finishing model training 
    logging_info('')
    logging_info('Finished the inference of the model YOLO v8' + LINE_FEED)

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())


# ###########################################
# Methods of Level 2
# ###########################################

def get_parameters(full_path_project, parameters_filename):
    '''
    Get dictionary parameters for processing
    '''    
    # getting parameters 
    path_and_parameters_filename = os.path.join(full_path_project, parameters_filename)
    parameters = Utils.read_json_parameters(path_and_parameters_filename)

    # returning parameters 
    return parameters

def set_input_image_folders(parameters):
    '''
    Set folder name of input images dataset
    '''    
    
    # getting image dataset folder according processing parameters 
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    image_dataset_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['input']['input_dataset']['input_dataset_path'],
        parameters['input']['input_dataset']['annotation_format'],
        input_image_size + 'x' + input_image_size,
    )

    # setting image dataset folder in processing parameters 
    parameters['processing']['image_dataset_folder'] = image_dataset_folder
    parameters['processing']['image_dataset_folder_train'] = \
        os.path.join(image_dataset_folder, 'train')
    parameters['processing']['image_dataset_folder_valid'] = \
        os.path.join(image_dataset_folder, 'valid')
    parameters['processing']['image_dataset_folder_test'] = \
        os.path.join(image_dataset_folder, 'test')

def get_running_id(parameters):
    '''
    Get last running id to calculate the current id
    '''    

    # setting control filename 
    running_control_filename = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['running_control_filename'],
    )

    # getting control info 
    running_control = Utils.read_json_parameters(running_control_filename)

    # calculating the current running id 
    running_control['last_running_id'] = int(running_control['last_running_id']) + 1

    # updating running control file 
    running_id = int(running_control['last_running_id'])

    # saving file 
    Utils.save_text_file(running_control_filename, \
                         Utils.get_pretty_json(running_control), 
                         NEW_FILE)

    # updating running id in the processing parameters 
    parameters['processing']['running_id'] = running_id

    # returning the current running id
    # return running_id

def set_result_folders(parameters):
    '''
    Set folder name of output results
    '''

    # creating results folders 
    main_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['inference_results']['main_folder']
    )
    parameters['inference_results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating model folder 
    parameters['inference_results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        main_folder,
        parameters['inference_results']['model_folder']
    )
    parameters['inference_results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        model_folder,
        parameters['inference_results']['action_folder']
    )
    parameters['inference_results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['inference_results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        action_folder,
        parameters['inference_results']['running_folder']
    )
    parameters['inference_results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['inference_results']['processing_parameters_folder']
    )
    parameters['inference_results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    weights_folder = os.path.join(
        running_folder,
        parameters['inference_results']['weights_folder']
    )
    parameters['inference_results']['weights_folder'] = weights_folder
    Utils.create_directory(weights_folder)

    metrics_folder = os.path.join(
        running_folder,
        parameters['inference_results']['metrics_folder']
    )
    parameters['inference_results']['metrics_folder'] = metrics_folder
    Utils.create_directory(metrics_folder)

    inferenced_image_folder = os.path.join(
        running_folder,
        parameters['inference_results']['inferenced_image_folder']
    )
    parameters['inference_results']['inferenced_image_folder'] = inferenced_image_folder
    Utils.create_directory(inferenced_image_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['inference_results']['log_folder']
    )
    parameters['inference_results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

    results_folder = os.path.join(
        running_folder,
        parameters['inference_results']['results_folder']
    )
    parameters['inference_results']['results_folder'] = results_folder
    Utils.create_directory(results_folder)

def get_device(parameters):
    '''
    Get device CUDA to train models
    '''    

    logging_info(f'')
    logging_info(f'>> Get device')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parameters['processing']['device'] = f'{device}'
    
    logging_info(f'Device: {device}')
    
    # returning current device 
    return device 

def save_processing_parameters(parameters_filename, parameters):
    '''
    Update parameters file of the processing
    '''

    logging_info(f'')
    logging_info(f'>> Save processing parameters of this running')

    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['inference_results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

def copy_weights_file(parameters):
    '''
    Copying weights file to inference step
    '''

    logging_info(f'')
    logging_info(f'>> Copy weights file of the model for inference')
    logging_info(f"Folder name: {parameters['input']['inference']['weights_folder']}")
    logging_info(f"Filename   : {parameters['input']['inference']['weights_filename']}")

    Utils.copy_file_same_name(
        parameters['input']['inference']['weights_filename'],
        parameters['input']['inference']['weights_folder'],
        parameters['inference_results']['weights_folder']
    )

def get_neural_network_model(parameters, device):
    '''
    Get neural network model
    '''

    logging_info(f'')
    logging_info(f'>> Get neural network model')

    # Load a YOLO model with the pretrained weights
    path_and_yolo_model_filename_with_best_weights = os.path.join(
        parameters['input']['inference']['weights_folder'],
        parameters['input']['inference']['weights_filename'],
    )
    logging_info(f'Model used with the best weights of the training step: {path_and_yolo_model_filename_with_best_weights}')

    model = YOLO(path_and_yolo_model_filename_with_best_weights)

    logging.info(f'{model}')

    # returning neural network model
    return model

# getting statistics of input dataset 
def get_input_dataset_statistics(parameters):
    
    annotation_statistics = AnnotationsStatistic()
    annotation_statistics.processing_statistics(parameters)
    return annotation_statistics
    
def show_input_dataset_statistics(annotation_statistics):

    logging_info(f'Input dataset statistic')
    logging_info(annotation_statistics.to_string())

# def show_number_of_images_for_inference(parameters):

#     logging_info(f'')
#     logging_info(f'>> Show number of test images')

#      # getting list of test images for inference 
#     input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
#     test_image_dataset_folder = os.path.join(
#         parameters['processing']['research_root_folder'],
#         parameters['input']['input_dataset']['input_dataset_path'],
#         parameters['input']['input_dataset']['annotation_format'],
#         input_image_size + 'x' + input_image_size,
#         'test'
#     )

#     test_images = glob.glob(f"{test_image_dataset_folder}/*.jpg")

#     logging_info(f"Number of test images: {len(test_images)}")   

def inference_neural_network_model(parameters, device, model):
    '''
    Execute inference of the neural network model
    '''

    # Run batched inference on a list of images   
    image_dataset_folder_test_images = os.path.join(
        parameters['processing']['image_dataset_folder_test'],
        'images',
    )
    image_dataset_folder_test_labels = os.path.join(
        parameters['processing']['image_dataset_folder_test'],
        'labels',
    )

    logging_info(f"Number of test images: {len(image_dataset_folder_test_images)}")   

    # get list of all test images for inference 
    test_images = Utils.get_files_with_extensions(image_dataset_folder_test_images, '.jpg')
    test_images_with_path = []
    for test_image in test_images:
        test_image = os.path.join(
            image_dataset_folder_test_images,
            test_image
        )
        test_images_with_path.append(test_image)
    
    # getting annotation files 
    test_labels = Utils.get_files_with_extensions(image_dataset_folder_test_labels, '.txt')
    test_labels_with_path = []
    for test_label in test_labels:
        test_label = os.path.join(
            image_dataset_folder_test_images,
            test_label
        )
        test_labels_with_path.append(test_label)

    # logging_info(f'test_images: {test_images_with_path}')
    # logging_info(f'test_labels: {test_labels_with_path}')

    # results = model(test_images)  # return a list of Results objects
    # logging_info(f'results: {results}')

    count = 0
    inference_results = []
    for test_image in test_images_with_path:
        count += 1
        logging_info(f'-'*70)
        logging_info(f'test_image #{count} {test_image}')
        results = model(source=test_image, 
                        show=False,
                        project=parameters['inference_results']['results_folder'],
                        save=True,
                        save_conf=True
                        )
        inference_results.append(results)
        logging_info(f'results: {results}')
        # break

# !yolo task=detect mode=predict model={best_weights} conf={WHITE_MOLD_THRESHOLD} 
#       source={local_image_dataset_test_images_path} save=True save_conf=True

    # # Process results list
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     result.show()  # display to screen
    #     result.save(filename='result.jpg')  # save to disk


# ###########################################
# Methods of Level 3
# ###########################################

# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
