from cral.models.semantic_segmentation import annotate_image
from cral.pipeline.semantic_segmentation_pipeline import SemanticSegPipe
from IPython.display import display

new_pipe = SemanticSegPipe()
new_pipe.dcrf = True #enable or disable dcrf

pred_func = new_pipe.prediction_model('./tmp/deeplab_final/')
# pred_func = new_pipe.prediction_model('./outm/')

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
image_path = 'axa.jpg'

mask_array = pred_func(image_path)
segmented_image = annotate_image(image_path, mask_array)

segmented_image.save("img1.png")

# from cral.models.semantic_segmentation import SparseMeanIoU
# model = tf.keras.models.load_model('./outm/')

# model = tf.keras.models.load_model('./tmp/deeplab_final/',compile=False,custom_objects={'SparseMeanIoU': SparseMeanIoU})
# model.compile(metrics=['accuracy'])
# model.save('outm')