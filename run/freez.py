# in this case, we save the frozen model to one single .pb model file
# and load it for forward inference

# for save model, first import meta graph and restore variables for checkpoint file
#     and then send the GraphDef to graph_util.convert_variables_to_constants and get
#     a new GraphDef object which is a simplified version of original one,
#     and then serialize the GraphDef and write it to disk using tf.gfile
#
# for load model, first we read the .pb file from disk and deserialize it,as a return we
#     get a GraphDef object, and then import it into the default graph use import_graph_def,
#     then we can get tensor from the graph for inference.
#
import tensorflow as tf
## use tfv1 for conversion
if tf.__version__.startswith("2."):
    tfv1 = tf.compat.v1
tfv1.disable_eager_execution()
from tensorflow.compat.v1.graph_util import convert_variables_to_constants


def save():
    model = tf.keras.models.load_model('dl-final',compile=False)
    model.summary()
    ## get_session is deprecated in tf2
    tfsession = tfv1.keras.backend.get_session()
    print([node.op.name for node in model.outputs])
    #['output_layer/BiasAdd']
    output_graph_def = convert_variables_to_constants(tfsession, tfsession.graph_def, [node.op.name for node in model.outputs])
    output_graph_def_filename = './frozen_model.pb'
    with tf.io.gfile.GFile(output_graph_def_filename, 'wb')as f:
        f.write(output_graph_def.SerializeToString())

def load():
    frozen_model_name = './frozen_model.pb'
    with tf.io.gfile.GFile(frozen_model_name, 'rb') as f:
        restored_graph_def = tf.compat.v1.GraphDef()
        restored_graph_def.ParseFromString(f.read())
        if restored_graph_def is None:
          raise RuntimeError('Cannot find inference graph in tar archive.')
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def=restored_graph_def,name='')
        input_tensor = graph.get_tensor_by_name('input_1:0')
        conv_tensor = graph.get_tensor_by_name('output_layer/BiasAdd:0')
        print(input_tensor)
        print(conv_tensor)
if __name__ == '__main__':
    # save()
    load()