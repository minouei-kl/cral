from cral.pipeline.semantic_segmentation_pipeline import SemanticSegPipe
import os
import tensorflow as tf

new_pipe = SemanticSegPipe()

"""# add data and create tfrecords"""
DATA_DIR = "/netscratch/minouei/versicherung/version2"

new_pipe.add_data(
    train_images_dir=os.path.join(DATA_DIR, 'images/train'),
    train_anno_dir=os.path.join(DATA_DIR, 'annotations/train'),
    annotation_format='grayscale',
    split=0.001)

new_pipe.lock_data()

"""# set up Deeplabv3+"""

from cral.models.semantic_segmentation.deeplabv3 import Deeplabv3Config

deeplab_config = Deeplabv3Config(
    height=576,
    width=576, output_stride = 8)
slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)

new_pipe.set_algo(
    feature_extractor='resnet50',base_trainable=True,
    config=deeplab_config,distribute_strategy=mirrored_strategy
)



"""# train"""

# print(new_pipe.model.summary())

new_pipe.train(
    num_epochs=8,
    batch_size=8,
    snapshot_prefix='deeplab',
    snapshot_path='./tmp/',
    snapshot_every_n=1,distribute_strategy=mirrored_strategy
    )
