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
# from ultralytics.utils.metrics.ConfusionMatrix import *
# from ultralytics.utils.Metric import *
from IPython.display import display, Image

# torchvision libraries
import torch

# these are the helper libraries imported.

# Importing python modules
from common.manage_log import *
from common.tasks import Tasks
from common.entity.ImageAnnotation import ImageAnnotation
from common.entity.AnnotationsStatistic import AnnotationsStatistic
from common.metrics import *
from create_yaml_file import * 

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
        parameters['test_results']['log_folder'], 
        parameters['test_results']['log_filename']
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

    # creating yaml file with parameters used by Ultralytics
    processing_tasks.start_task('Creating yaml file for ultralytics')
    create_yaml_file_for_ultralytics(parameters)
    processing_tasks.finish_task('Creating yaml file for ultralytics')

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
    if parameters['processing']['show_statistics_of_input_dataset']:
        processing_tasks.start_task('Getting statistics of input dataset')
        annotation_statistics = get_input_dataset_statistics(parameters)
        show_input_dataset_statistics(parameters, annotation_statistics)
        processing_tasks.finish_task('Getting statistics of input dataset')    

    # inference the neural netowrk model
    processing_tasks.start_task('Running prediction on the test images dataset')
    # test_neural_network_model(parameters, device, model)
    test_neural_network_model_ultralytics(parameters, device, model)
    processing_tasks.finish_task('Running prediction on the test images dataset')

    # showing input dataset statistics
    # if parameters['processing']['show_statistics_of_input_dataset']:
    #     show_input_dataset_statistics(parameters, annotation_statistics)

    # finishing model training 
    logging_info('')
    logging_info('Finished the inference of the model YOLO v8' + LINE_FEED)

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())

    # copying processing files to log folder 
    # copy_processing_files_to_log(parameters)


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
        parameters['test_results']['main_folder']
    )
    parameters['test_results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating model folder 
    parameters['test_results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        main_folder,
        parameters['test_results']['model_folder']
    )
    parameters['test_results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        model_folder,
        parameters['test_results']['action_folder']
    )
    parameters['test_results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['test_results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        action_folder,
        parameters['test_results']['running_folder']
    )
    parameters['test_results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['test_results']['processing_parameters_folder']
    )
    parameters['test_results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    weights_folder = os.path.join(
        running_folder,
        parameters['test_results']['weights_folder']
    )
    parameters['test_results']['weights_folder'] = weights_folder
    Utils.create_directory(weights_folder)

    metrics_folder = os.path.join(
        running_folder,
        parameters['test_results']['metrics_folder']
    )
    parameters['test_results']['metrics_folder'] = metrics_folder
    Utils.create_directory(metrics_folder)

    inferenced_image_folder = os.path.join(
        running_folder,
        parameters['test_results']['inferenced_image_folder']
    )
    parameters['test_results']['inferenced_image_folder'] = inferenced_image_folder
    Utils.create_directory(inferenced_image_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['test_results']['log_folder']
    )
    parameters['test_results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

    results_folder = os.path.join(
        running_folder,
        parameters['test_results']['results_folder']
    )
    parameters['test_results']['results_folder'] = results_folder
    Utils.create_directory(results_folder)

def create_yaml_file_for_ultralytics(parameters):

    # preparing parameters 
    yolo_v8_yaml_filename = parameters['processing']['yolo_v8_yaml_filename']
    path_and_filename_white_mold_yaml = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        yolo_v8_yaml_filename
    )
    image_dataset_folder = parameters['processing']['image_dataset_folder']
    classes = (parameters['neural_network_model']['classes'])[:5]
    
    # creating yaml file 
    create_project_yaml_file_for_test(
        path_and_filename_white_mold_yaml,
        image_dataset_folder,
        classes,    
    )

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
        parameters['test_results']['processing_parameters_folder'], 
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
        parameters['test_results']['weights_folder']
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

    logging.info(f'model: {model}')
    logging.info(f'model.info(): {model.info()}')

    # returning neural network model
    return model

# getting statistics of input dataset 
def get_input_dataset_statistics(parameters):
    
    annotation_statistics = AnnotationsStatistic()
    # steps = ['train', 'valid', 'test'] 
    steps = ['test'] 
    annotation_statistics.processing_statistics(parameters, steps)
    return annotation_statistics

def show_input_dataset_statistics(parameters, annotation_statistics):

    logging_info(f'Input dataset statistic')
    logging_info(annotation_statistics.to_string())
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + '_annotations_statistics.xlsx',
    )
    annotation_format = parameters['input']['input_dataset']['annotation_format']
    input_image_size = parameters['input']['input_dataset']['input_image_size']
    classes = (parameters['neural_network_model']['classes'])[1:5]
    annotation_statistics.save_annotations_statistics(
        path_and_filename,
        annotation_format,
        input_image_size,
        classes
    )

def test_neural_network_model(parameters, device, model):
    '''
    Execute inference of the neural network model
    Confusion matrix of YOLOv8:
    - https://medium.com/@a0922/confusion-matrix-of-yolov8-97fd7ff0074e
    '''

    logging_info(f'')
    logging_info(f'Testing images')
    logging_info(f'')

    # getting classes 
    classes =  parameters['neural_network_model']['classes']

    # Run batched inference on a list of images   
    image_dataset_folder_test_images = os.path.join(
        parameters['processing']['image_dataset_folder_test'],
        'images',
    )
    image_dataset_folder_test_labels = os.path.join(
        parameters['processing']['image_dataset_folder_test'],
        'labels',
    )
    logging_info(f'Test image dataset folder: {image_dataset_folder_test_images}')
    logging_info(f'')

    # get list of all test images for inference 
    test_images = Utils.get_files_with_extensions(image_dataset_folder_test_images, '.jpg')
    test_images_with_path = []
    for test_image in test_images:
        test_image = os.path.join(
            image_dataset_folder_test_images,
            test_image
        )
        test_images_with_path.append(test_image)
    
    data_file_yaml = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['yolo_v8_yaml_filename']
    )

    # creating metric object 
    inference_metric = Metrics(model=parameters['neural_network_model']['model_name'])

    # logging_info(f'len test_image #{len(test_images_with_path)}')

    count = 0    
    for test_image in test_images_with_path:
        count += 1

        results = model.predict(
            data=data_file_yaml, 
            source=test_image, 
            imgsz=parameters['input']['input_dataset']['input_image_size'],
            project=parameters['test_results']['results_folder'],
            conf=parameters['neural_network_model']['threshold'],
            iou=parameters['neural_network_model']['iou_threshold'],
            device=device,
            verbose=True,
            show=False,
            save=True,
            save_conf=True,
            plots=True,
        )

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            names = result.names  
            result.show()  # display to screen
            result.save(filename='result.jpg')  # save to disk

            # logging_info(f'result: {result}')
            logging_info(f'boxes: {boxes}')
            logging_info(f'boxes.data: {boxes.data}')
            # logging_info(f'probs: {probs}')
            logging_info(f'names: {names}')
            
        continue 



        # results = model(data=data_file_yaml, 
        #                 source=test_image, 
        #                 verbose=True, 
        #                 show=False,
        #                 project=parameters['test_results']['results_folder'],
        #                 save=True,
        #                 save_conf=True
        #                 )
        # results = model(source=test_image, 
        #                 show=False,
        #                 project=parameters['test_results']['results_folder'],
        #                 save=True,
        #                 save_conf=True
        #                 )
        
        # extracting parts of path and image filename 
        path, filename_with_extension, filename, extension = Utils.get_filename(test_image)

        # setting the annotation filename 
        path_and_filename_yolo_annotation = os.path.join(
            image_dataset_folder_test_labels, 
            filename + '.txt'
            )

        # logging_info(f'-'*70)
        logging_info(f'Test image #{count} {filename_with_extension}')
        # logging_info(f'test_label #{count} {path_and_filename_yolo_annotation}')

        # getting all annotations of the image 
        image_annotation = ImageAnnotation()
        height = width = parameters['input']['input_dataset']['input_image_size']
        image_annotation.get_annotation_file_in_yolo_v5_format(
            path_and_filename_yolo_annotation, classes, height, width
        )
        # logging_info(f'image_annotation: {image_annotation.to_string()}')

        # getting target bounding boxes 
        targets = image_annotation.get_tensor_target(classes)
        # logging_info(f'target annotated: {targets}')

        # setting target and predicted bounding boxes for metrics 
        # new_targets = []
        # item_target = {
        #     "boxes": target['boxes'],
        #     "labels": target['labels']
        #     }
        # new_targets.append(item_target)

        new_predicteds = []
        for result in results:
            result = result.to('cpu')
            for box in result.boxes:    
                # logging_info(f'boxes predicted: {box.xyxy}')

                item_predicted = {
                    "boxes": box.xyxy,
                    "scores": box.conf,
                    "labels": torch.tensor(box.cls, dtype=torch.int),
                    }
                new_predicteds.append(item_predicted)

        # logging_info(f'targets: {targets}')
        # logging_info(f'new_predicteds: {new_predicteds}')

        # setting target and predicted bounding boxes for metrics
        inference_metric.set_details_of_inferenced_image(
            filename_with_extension, targets, new_predicteds) 
        # inference_metric.target.extend(target)
        # inference_metric.preds.extend(new_predicteds)
        # logging_info(f'inference_metric.to_string: {inference_metric.to_string()}')
        # logging_info(f'--------------------------------------------------')

        # # Process results list
        # for result in results:
        #     logging_info(f'-'*50)
        #     logging_info(f'result: {result}')
        #     logging_info(f'result.boxes: {result.boxes}')
        #     logging_info(f'result.masks: {result.masks}')
        #     logging_info(f'result.probs: {result.probs}')
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     result.show()  # display to screen
        #     result.save(filename='result.jpg')  # save to disk

    # Computing Confusion Matrix 
    model_name = parameters['neural_network_model']['model_name']
    num_classes = 5
    threshold = parameters['neural_network_model']['threshold']
    iou_threshold = parameters['neural_network_model']['iou_threshold']
    metrics_folder = parameters['test_results']['metrics_folder']
    inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold, metrics_folder)
    inference_metric.confusion_matrix_to_string()

    # saving confusion matrix plots 
    title =  'Full Confusion Matrix' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + '  # bounding box -' + \
             ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
             '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
             '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])

    # logging_info(f'Bounding boxes target                : ' + \
    #              f'{self.confusion_matrix_summary["Numsey$2023number_of_bounding_boxes_target"]}')
    # logging_info(f'Bounding boxes predicted             : ' + \
    #              f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted"]}')
    # logging_info(f'Bounding boxes predicted with target : ' + \
    #              f'{self.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]}')
    # logging_info(f'Number of ghost preditions           : ' + \
    #              f'{self.confusion_matrix_summary["number_of_ghost_predictions"]}')
    # logging_info(f'Number of undetected objects         : ' + \
    #              f'{self.confusion_matrix_summary["number_of_undetected_objects"]}')

    path_and_filename = os.path.join(parameters['test_results']['metrics_folder'], 
        'confusion_matrix_full_' + parameters['neural_network_model']['model_name'] + '.png'
    )
    cm_classes = classes[0:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Ghost predictions')    
    y_labels_names.append('Undetected objects')
    format='.0f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_full.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix,
                                      path_and_filename, 
                                      x_labels_names, y_labels_names)                                    
        
    title = 'Confusion Matrix'
    path_and_filename = os.path.join(parameters['test_results']['metrics_folder'], 
        'confusion_matrix_' + parameters['neural_network_model']['model_name'] + '.png'
    )
    cm_classes = classes[1:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    format='.0f'
    Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix,
                                      path_and_filename,
                                      x_labels_names, y_labels_names)                      

    title =  'Full Confusion Matrix Normalized' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + '  # bounding box -' + \
             ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
             '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
             '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_full_normalized.png'
    )
    cm_classes = classes[0:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Ghost predictions')    
    y_labels_names.append('Undetected objects')
    format='.2f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix_normalized, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_full_normalized.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix_normalized,
                                      path_and_filename,
                                      x_labels_names, y_labels_names)                      

    title =  'Confusion Matrix Normalized' + \
             ' - Model: ' + parameters['neural_network_model']['model_name'] + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + '  # bounding box -' + \
             ' predicted with target: ' + str(inference_metric.confusion_matrix_summary['number_of_bounding_boxes_predicted_with_target']) + \
             '   ghost predictions: ' + str(inference_metric.confusion_matrix_summary['number_of_ghost_predictions']) + \
             '   undetected objects: ' + str(inference_metric.confusion_matrix_summary['number_of_undetected_objects'])
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_normalized.png'
    )
    cm_classes = classes[1:5]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    format='.2f'
    Utils.save_plot_confusion_matrix(inference_metric.confusion_matrix_normalized, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'], 
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_normalized.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.confusion_matrix_normalized,
                                      path_and_filename,
                                      x_labels_names, y_labels_names)

    # saving metrics from confusion matrix
    path_and_filename = os.path.join(
        parameters['test_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + '_confusion_matrix_metrics.xlsx'
    )
    Utils.save_metrics_from_confusion_matrix_excel(
        path_and_filename, 
        parameters['neural_network_model']['model_name'],
        inference_metric.get_model_accuracy(),
        inference_metric.get_model_precision(),
        inference_metric.get_model_recall(),
        inference_metric.get_model_f1_score(),
        inference_metric.get_model_specificity(),
        inference_metric.get_model_dice(),
        inference_metric.confusion_matrix_summary["number_of_images"],
        inference_metric.confusion_matrix_summary["number_of_bounding_boxes_target"],
        inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted"],
        inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"],
        inference_metric.confusion_matrix_summary["number_of_ghost_predictions"],
        inference_metric.confusion_matrix_summary["number_of_undetected_objects"], 
    )
    
    # get performance metrics 
    logging_info(f'')    
    logging_info(f"Performance Metrics of model {parameters['neural_network_model']['model_name']}")
    logging_info(f'')
    model_accuracy = inference_metric.get_model_accuracy()
    logging_info(f'accuracy    : {model_accuracy:.4f}')
    model_precision = inference_metric.get_model_precision()
    logging_info(f'precision   : {model_precision:.4f}')
    model_recall = inference_metric.get_model_recall()
    logging_info(f'recall      : {model_recall:.4f}')
    model_f1_score = inference_metric.get_model_f1_score()
    logging_info(f'f1-score    : {model_f1_score:.4f}')
    model_specificity = inference_metric.get_model_specificity()
    logging_info(f'specificity : {model_specificity:.4f}')
    model_dice = inference_metric.get_model_dice()
    logging_info(f'dice        : {model_dice:.4f}')
    logging_info(f'')



def test_neural_network_model_ultralytics(parameters, device, model):
    '''
    Execute inference of the neural network model
    Confusion matrix of YOLOv8:
    - https://medium.com/@a0922/confusion-matrix-of-yolov8-97fd7ff0074e
    '''

    logging_info(f'')
    logging_info(f'Testing images')
    logging_info(f'')

    # getting classes 
    classes =  parameters['neural_network_model']['classes']

    # Run batched inference on a list of images   
    image_dataset_folder_test_images = os.path.join(
        parameters['processing']['image_dataset_folder_test'],
        'images',
    )
    image_dataset_folder_test_labels = os.path.join(
        parameters['processing']['image_dataset_folder_test'],
        'labels',
    )
    logging_info(f'Test image dataset folder: {image_dataset_folder_test_images}')
    logging_info(f'')

    # get list of all test images for inference 
    test_images = Utils.get_files_with_extensions(image_dataset_folder_test_images, '.jpg')
    test_images_with_path = []
    for test_image in test_images:
        test_image = os.path.join(
            image_dataset_folder_test_images,
            test_image
        )
        test_images_with_path.append(test_image)
    
    data_file_yaml = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['yolo_v8_yaml_filename']
    )

    # creating metric object 
    # inference_metric = Metrics(model=parameters['neural_network_model']['model_name'])

    # logging_info(f'len test_image #{len(test_images_with_path)}')

    count = 0    
    # for test_image in test_images_with_path:
    #     count += 1
    #     logging_info(f'model.predict test_image #{count} {test_image}')

    #     results = model.predict(
    #         data=data_file_yaml, 
    #         source=test_image, 
    #         imgsz=parameters['input']['input_dataset']['input_image_size'],
    #         project=parameters['test_results']['results_folder'],
    #         conf=parameters['neural_network_model']['threshold'],
    #         iou=parameters['neural_network_model']['iou_threshold'],
    #         device=device,
    #         verbose=True,
    #         show=False,
    #         save=True,
    #         save_conf=True,
    #         plots=True,
    #     )

    #     for result in results:
    #         boxes = result.boxes  # Boxes object for bounding box outputs
    #         masks = result.masks  # Masks object for segmentation masks outputs
    #         keypoints = result.keypoints  # Keypoints object for pose outputs
    #         probs = result.probs  # Probs object for classification outputs
    #         names = result.names  
    #         result.show()  # display to screen
    #         result.save(filename='result.jpg')  # save to disk

    #         # logging_info(f'result: {result}')
    #         logging_info(f'boxes: {boxes}')
    #         logging_info(f'boxes.data: {boxes.data}')
    #         # logging_info(f'probs: {probs}')
    #         logging_info(f'names: {names}')
            

    logging_info(f'-------------------------------------')

    # results = model.predict(
    #     data=data_file_yaml,
    #     source=test_images_with_path,
    #     imgsz=parameters['input']['input_dataset']['input_image_size'],
    #     project=parameters['test_results']['results_folder'],
    #     conf=parameters['neural_network_model']['threshold'],
    #     iou=parameters['neural_network_model']['iou_threshold'],
    #     device=device,
    #     verbose=True,
    #     show=False,
    #     save=True,
    #     save_conf=True,
    #     plots=True,
    # )

    # logging_info(f'len(results): {len(results)}')
    # # logging_info(f'results: {results}')

    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     names = result.names  

    #     logging_info(f'result.path: {result.path}')
    #     logging_info(f'boxes: {boxes}')
    #     logging_info(f'boxes.data: {boxes.data}')
    #     logging_info(f'names: {names}')
            
    logging_info(f'-------------------------------------')

    # for test_image in test_images_with_path:
    #     count += 1
    #     logging_info(f'model.val test_image #{count} {test_image}')
    #     # source=test_image, 
    #     metric_results = model.val(
    #         data=data_file_yaml, 
    #         imgsz=parameters['input']['input_dataset']['input_image_size'],
    #         project=parameters['test_results']['results_folder'],
    #         conf=parameters['neural_network_model']['threshold'],
    #         iou=parameters['neural_network_model']['iou_threshold'],
    #         device=device,
    #         verbose=True,
    #         show=False,
    #         save=True,
    #         save_conf=True,
    #         plots=True,
    #     )

    #     logging_info(f'metric_results.box: {metric_results.box}')
    #     logging_info(f'metric_results.confusion_matrix: {metric_results.confusion_matrix}')
    #     logging_info(f'metric_results.results_dict: {metric_results.results_dict}')
          
    count += 1
    # logging_info(f'model.val test_image #{count} {test_image}')
    metric_results = model.val(
        data=data_file_yaml, 
        imgsz=parameters['input']['input_dataset']['input_image_size'],
        project=parameters['test_results']['results_folder'],
        conf=parameters['neural_network_model']['threshold'],
        iou=parameters['neural_network_model']['iou_threshold'],
        device=device,
        verbose=True,
        show=False,
        save=True,
        save_conf=True,
        plots=True,
    )

    logging_info(f'metric_results.box: {metric_results.box}')
    logging_info(f'metric_results.confusion_matrix: {metric_results.confusion_matrix}')
    logging_info(f'metric_results.confusion_matrix.matrix: {metric_results.confusion_matrix.matrix}')
    logging_info(f'metric_results.results_dict: {metric_results.results_dict}')
    logging_info(f'=====================================================================')
    logging_info(f'metric_results.box.map: {metric_results.box.map}')
    logging_info(f'metric_results.box.map50: {metric_results.box.map50}')
    logging_info(f'metric_results.box.map75: {metric_results.box.map75}')
    logging_info(f'metric_results.box.maps: {metric_results.box.maps}')

def copy_processing_files_to_log(parameters):
    input_path = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
    )
    output_path = parameters['test_results']['log_folder']
    input_filename = output_filename = 'yolo_v8_inference_errors'
    Utils.copy_file(input_filename, input_path, output_filename, output_path)

    input_filename = output_filename = 'yolo_v8_inference_output'
    Utils.copy_file(input_filename, input_path, output_filename, output_path)


# ###########################################
# Methods of Level 3
# ###########################################

# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
