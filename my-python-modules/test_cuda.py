import torch
import os

from ultralytics import YOLO
from torchinfo import summary
from ptflops import get_model_complexity_info

if __name__ == '__main__':
    print(f'Test CUDA')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')   
    print(f'device: {device}')

    print(f'')
    print(f'Showing Model size')
    print(f'')

    # getting model 
    input_inference_weights_folder = "/home/lovelace/proj/proj939/rubenscp/research/white-mold-inference-weights/exp-008-training-300x300-merging-classes-balanced-image-all-classes"
    input_inference_weights_filename = "yolov10x.pt-running-0053-300x300.pt"    
    path_and_yolo_model_filename_with_best_weights = os.path.join(
        input_inference_weights_folder, input_inference_weights_filename
    )
    model = YOLO(path_and_yolo_model_filename_with_best_weights)

    # Get the model summary to see size in MB and parameter details
    summary(model, input_size=(1, 3, 224, 224))  # Adjust input_size based on your input
    
    # Calculate GFLOPS and parameters
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
        print(f'')
        print(f"GFLOPS: {flops}, Parameters: {params}")
        print(f'')
