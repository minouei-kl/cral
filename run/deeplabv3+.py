from cral.pipeline.semantic_segmentation_pipeline import SemanticSegPipe
import os

new_pipe = SemanticSegPipe()

"""# add data and create tfrecords"""
DATA_DIR = "/home/minouei/Downloads/datasets/contract/version2"

new_pipe.add_data(
    train_images_dir=os.path.join(DATA_DIR, 'images/train'),
    train_anno_dir=os.path.join(DATA_DIR, 'annotations/train'),
    annotation_format='grayscale',
    split=0.002)

new_pipe.lock_data()

"""# set up Deeplabv3+"""

from cral.models.semantic_segmentation.deeplabv3 import Deeplabv3Config

deeplab_config = Deeplabv3Config(
    height=576, 
    width=576, output_stride = 8)

new_pipe.set_algo(
    feature_extractor='resnet50',base_trainable=True,
    config=deeplab_config)

deeplab_config

"""# train"""

# print(new_pipe.model.summary())

new_pipe.train(
    num_epochs=10,
    snapshot_prefix='deeplab',
    snapshot_path='./tmp/',
    snapshot_every_n=1
    )

"""# predict"""

# from cral.models.semantic_segmentation import annotate_image
# from cral.pipeline.semantic_segmentation_pipeline import SemanticSegPipe
# from IPython.display import display

# new_pipe = SemanticSegPipe()
# new_pipe.dcrf = True #enable or disable dcrf

# pred_func = new_pipe.prediction_model('./deeplab_resnet50/')

# import matplotlib.pyplot as plt
# import cv2
# import numpy as np

# image_path = './seg_data/mini_ADE20K/images/ADE_val_00000011.jpg'

# mask_array = pred_func(image_path)
# segmented_image = annotate_image(image_path, mask_array)

# display(segmented_image)