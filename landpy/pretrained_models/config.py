import os

MODEL_URLS = {
    'face_tracker': 'https://github.com/bateni1380/pretrained_models/raw/refs/heads/main/model_files/face_detection_short_range.tflite',
    'face_landmark_detector': 'https://github.com/bateni1380/pretrained_models/raw/refs/heads/main/model_files/face_landmark.tflite',
    'expression_detector': 'https://github.com/bateni1380/pretrained_models/raw/refs/heads/main/model_files/facial_expression.h5',
    'face_segmentation_detector': 'https://github.com/bateni1380/pretrained_models/raw/refs/heads/main/model_files/face_segmentation.tflite'
}   

DOWNLOAD_DIR = os.path.expanduser('~/.keras/models/')
