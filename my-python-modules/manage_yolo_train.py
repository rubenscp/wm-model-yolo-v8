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
import pandas as pd

# torchvision libraries
import torch

# # these are the helper libraries imported.
# from engine import train_one_epoch, evaluate

# Import python code from White Mold Project 
from common.manage_log import *
from common.tasks import Tasks
from common.entity.AnnotationsStatistic import AnnotationsStatistic
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
    Main method that perform training of the neural network model.

    All values of the parameters used here are defined in the external file "wm_model_yolo_v8_parameters.json".

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
    set_results_folder(parameters)
    processing_tasks.finish_task('Setting result folders')
    
    # creating log file 
    processing_tasks.start_task('Creating log file')
    logging_create_log(
        parameters['training_results']['log_folder'], parameters['training_results']['log_filename']
    )
    processing_tasks.finish_task('Creating log file')
    
    logging_info('White Mold Research')
    logging_info('Training the model YOLO v8' + LINE_FEED)

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

    # training neural netowrk model
    processing_tasks.start_task('Training neural netowrk model')
    model = train_yolo_v8_model(parameters, device, model)
    processing_tasks.finish_task('Training neural netowrk model')

    # saving the best weights file 
    processing_tasks.start_task('Saving best weights')
    save_best_weights(parameters)
    processing_tasks.finish_task('Saving best weights')

    # validating model 
    processing_tasks.start_task('Validating model')
    model = validate_yolo_v8_model(parameters, device, model)
    processing_tasks.finish_task('Validating model')
    
    # showing input dataset statistics
    # if parameters['processing']['show_statistics_of_input_dataset']:
    #     show_input_dataset_statistics(annotation_statistics)

    # finishing model training 
    logging_info('')
    logging_info('Finished the training of the model YOLOv8 ' + LINE_FEED)

    # creating plot of training loss
    create_plot_training_loss(parameters)    

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
    parameters['processing']['running_id_text'] = 'running-' + f'{running_id:04}'

    # returning the current running id
    return running_id

def set_results_folder(parameters):
    '''
    Set folder name of output results
    '''

    # resetting test results 
    parameters['test_results'] = {}

    # creating results folders 
    main_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['training_results']['main_folder']
    )
    parameters['training_results']['main_folder'] = main_folder
    Utils.create_directory(main_folder)

    # setting and creating model folder 
    parameters['training_results']['model_folder'] = parameters['neural_network_model']['model_name']
    model_folder = os.path.join(
        main_folder,
        parameters['training_results']['model_folder']
    )
    parameters['training_results']['model_folder'] = model_folder
    Utils.create_directory(model_folder)

    # setting and creating experiment folder
    experiment_folder = os.path.join(
        model_folder,
        parameters['input']['experiment']['id']
    )
    parameters['training_results']['experiment_folder'] = experiment_folder
    Utils.create_directory(experiment_folder)

    # setting and creating action folder of training
    action_folder = os.path.join(
        experiment_folder,
        parameters['training_results']['action_folder']
    )
    parameters['training_results']['action_folder'] = action_folder
    Utils.create_directory(action_folder)

    # setting and creating running folder 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    input_image_size = str(parameters['input']['input_dataset']['input_image_size'])
    parameters['training_results']['running_folder'] = running_id_text + "-" + input_image_size + 'x' + input_image_size   
    running_folder = os.path.join(
        action_folder,
        parameters['training_results']['running_folder']
    )
    parameters['training_results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating others specific folders
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['training_results']['processing_parameters_folder']
    )
    parameters['training_results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    weights_folder = os.path.join(
        running_folder,
        parameters['training_results']['weights_folder']
    )
    parameters['training_results']['weights_folder'] = weights_folder
    Utils.create_directory(weights_folder)

    # setting the base filename of weights
    weights_base_filename = parameters['neural_network_model']['model_name'] + '-' + \
                            running_id_text + "-" + input_image_size + 'x' + input_image_size + '.pt'
    parameters['training_results']['weights_base_filename'] = weights_base_filename

    metrics_folder = os.path.join(
        running_folder,
        parameters['training_results']['metrics_folder']
    )
    parameters['training_results']['metrics_folder'] = metrics_folder
    Utils.create_directory(metrics_folder)

    log_folder = os.path.join(
        running_folder,
        parameters['training_results']['log_folder']
    )
    parameters['training_results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

    results_folder = os.path.join(
        running_folder,
        parameters['training_results']['results_folder']
    )
    parameters['training_results']['results_folder'] = results_folder
    Utils.create_directory(results_folder)

    # setting folder of YOLO models 
    model_folder = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['neural_network_model']['model_folder'],
    )
    parameters['neural_network_model']['model_folder'] = model_folder
    
def create_yaml_file_for_ultralytics(parameters):

    # preparing parameters 
    yolo_v8_yaml_filename = parameters['processing']['yolo_v8_yaml_filename_train']
    path_and_filename_white_mold_yaml = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        yolo_v8_yaml_filename
    )
    image_dataset_folder = parameters['processing']['image_dataset_folder']
    number_of_classes = parameters['neural_network_model']['number_of_classes']
    classes = (parameters['neural_network_model']['classes'])[:(number_of_classes+1)]
    
    # creating yaml file 
    create_project_yaml_file_for_train_valid(
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
    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['training_results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

def get_neural_network_model(parameters, device):
    '''
    Get neural network model

    Upload the predefined model from Ultralytics repository:
    1) Donwload file in the Windows environment
    2) Copy file usinf VS Code
    > wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
    > wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt


    '''      
    
    logging_info(f'')
    logging_info(f'>> Get neural network model')

    # Load a YOLO model with the pretrained weights
    path_and_yolo_model_filename = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['neural_network_model']['model_folder'],
        parameters['neural_network_model']['model_name']
    )
    
    logging_info(f'Model used: {path_and_yolo_model_filename}')

    model = YOLO(path_and_yolo_model_filename)

    logging.info(f'{model}')

    # returning neural network model
    return model

# getting statistics of input dataset 
def get_input_dataset_statistics(parameters):
    
    annotation_statistics = AnnotationsStatistic()
    steps = ['train', 'valid', 'test'] 
    annotation_statistics.processing_statistics(parameters, steps)
    return annotation_statistics

def show_input_dataset_statistics(parameters, annotation_statistics):

    logging_info(f'Input dataset statistic')
    logging_info(annotation_statistics.to_string())
    path_and_filename = os.path.join(
        parameters['training_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + \
        '_' + parameters['processing']['running_id_text'] + '_annotations_statistics.xlsx',
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

def train_yolo_v8_model(parameters, device, model):
    '''
    Execute training of the neural network model
    '''

    logging_info(f'')
    logging_info(f'>> Training model')   

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # data_file_yaml = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-yolo-v8/white_mold_yolo_v8.yaml'
    data_file_yaml = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['yolo_v8_yaml_filename_train']
    )
    results = model.train(data=data_file_yaml, 
                          epochs=parameters['neural_network_model']['number_epochs'],
                          imgsz=parameters['input']['input_dataset']['input_image_size'],
                          project=parameters['training_results']['results_folder'],
                          conf=parameters['neural_network_model']['threshold'],
                          iou=parameters['neural_network_model']['iou_threshold_for_validation'],
                          lr0=parameters['neural_network_model']['learning_rate_initial'],
                          lrf=parameters['neural_network_model']['learning_rate_final'],
                          momentum=parameters['neural_network_model']['momentum'],
                          device=device,
                          pretrained=True,
                          verbose=True, 
                          show=True,
                          save=True,
                          plots=True
                          )

    logging_info(f'results: {results}')
    logging_info(f'Training of the model finished')

    # returing trained model
    return model 

def save_best_weights(parameters):

    # setting folders and filenames for weights file 
    input_filename  = 'best.pt'
    input_path      = os.path.join(parameters['training_results']['results_folder'], 'train/weights')
    output_filename =  parameters['training_results']['weights_base_filename']
    output_path     = parameters['training_results']['weights_folder']
    logging_info(f'input_filename: {input_filename}')
    logging_info(f'input_path: {input_path}')
    logging_info(f'output_filename: {output_filename}')
    logging_info(f'output_path: {output_path}')

    # copying weights file to be used in the inference step 
    Utils.copy_file(input_filename, input_path, output_filename, output_path)

def validate_yolo_v8_model(parameters, device, model):
    '''
    Execute validation in the model previously trained
    '''

    logging_info(f'')
    logging_info(f'>> Validating model')   

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # data_file_yaml = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-yolo-v8/white_mold_yolo_v8.yaml'    
    data_file_yaml = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['yolo_v8_yaml_filename_train']
    )
    metrics = model.val(
        data=data_file_yaml, 
        imgsz=parameters['input']['input_dataset']['input_image_size'],
        project=parameters['training_results']['results_folder'],
        conf=parameters['neural_network_model']['threshold'],
        iou=parameters['neural_network_model']['iou_threshold_for_validation'],
        device=device,
        verbose=True, 
        show=True,
        save=True,
        save_conf=True,
        plots=True,
    )

    # metrics = model.val(save_json=True, save_hybrid=True) 
    logging_info(f'metrics: {metrics}')
    # logging_info(f'box.map: {metrics.box.map}')
    # logging_info(f'box.map50: {metrics.box.map50}')
    # logging_info(f'box.map75: {metrics.box.map75}')
    # logging_info(f'box.maps: {metrics.box.maps}')
    
    logging_info(f'Validating of the model finished')
    
    # returing trained model
    return model

def copy_processing_files_to_log(parameters):
    input_path = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
    )
    output_path = parameters['training_results']['log_folder']    
    input_filename = output_filename = 'yolo_v8_train_errors'
    Utils.copy_file(input_filename, input_path, output_filename, output_path)

    input_filename = output_filename = 'yolo_v8_train_output'
    Utils.copy_file(input_filename, input_path, output_filename, output_path)


def create_plot_training_loss(parameters):
    '''
    Create plot of training loss 
    '''

    path_and_filename = os.path.join(
        parameters['training_results']['results_folder'],
        'train',
        'results.csv'
    )
    # path_and_filename = os.path.join(
    #     parameters['processing']['research_root_folder'],
    #     parameters['processing']['project_name_folder'],
    #     'results',
    #     'results.csv'
    # )
    logging_info(f'path_and_filename: {path_and_filename}')    

    # reading results file
    results_df = pd.read_csv(path_and_filename)

    # removing left spaces from the column name
    results_df.columns = results_df.columns.str.replace(' ','');

    # getting training losses
    epochs = results_df['epoch'].tolist()
    losses_per_epoch = results_df['train/box_loss'].tolist()
    train_loss_list_excel = []
    for epoch, loss in enumerate(losses_per_epoch):
        train_loss_list_excel.append([epoch+1, loss])

    # plot loss function for training
    path_and_filename = os.path.join(
        parameters['training_results']['metrics_folder'],     
        parameters['neural_network_model']['model_name'] + \
                    '_train_loss.png'
    )
    title = f'Training Loss for model {parameters["neural_network_model"]["model_name"]}'
    x_label = "Epochs"
    y_label = "Train Loss"
    logging_info(f'path_and_filename: {path_and_filename}')
    Utils.save_plot(losses_per_epoch, path_and_filename, title, x_label, y_label)

    # saving loss list to excel file
    logging_info(f'train_loss_list_excel final: {train_loss_list_excel}')    
    path_and_filename = os.path.join(
        parameters['training_results']['metrics_folder'],
        parameters['neural_network_model']['model_name'] + '_train_loss.xlsx'
    )
    logging_info(f'path_and_filename: {path_and_filename}')

    Utils.save_losses(train_loss_list_excel, path_and_filename)          

# ###########################################
# Methods of Level 3
# ###########################################

# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
