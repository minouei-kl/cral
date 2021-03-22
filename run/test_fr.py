 
import os
import cv2
import numpy as np
import tensorflow as tf
import operator

def find_corners_of_contour(cnt):
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in cnt]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in cnt]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in cnt]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in cnt]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [cnt[top_left][0], cnt[top_right][0], cnt[bottom_right][0], cnt[bottom_left][0]]

def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def crop_and_warp(img, crop_rect):
    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    top_left[0]=top_left[0]-3 if top_left[0]>3 else top_left[0]
    bottom_left[0]=bottom_left[0]-3 if bottom_left[0]>3 else bottom_left[0]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Get the longest side in the rectangle
    height = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
    ])
    width = max([
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    # Describe a Rectangle with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    return cv2.warpPerspective(img, m, (int(width), int(height)))


TOTAL_CLASSES = ['background', 'headerlogo', 'twocoltabel', 'recieveraddress',
                'text', 'senderaddress', 'ortdatum',
                 'companyinfo', 'fulltabletyp1', 'fulltabletyp2', 'copylogo',
                  'footerlogo', 'footertext', 'signatureimage', 'fulltabletyp3']

MODEL_CLASSES = TOTAL_CLASSES
N_CLASSES = 15
HEIGHT = 576
WIDTH = 576


def OverLayLabelOnImage(ImgIn, Label, W=0.6):
    # ImageIn is the image
    # Label is the label per pixel
    # W is the relative weight in which the labels will be marked on the image
    # Return image with labels marked over it
    Img = ImgIn.copy()
    TR = [0, 1, 0, 0,   0, 1, 1, 0, 0,   0.5, 0.7, 0.3, 0.5, 1,    0.5, 0.3]
    TB = [0, 0, 1, 0,   1, 0, 1, 0, 0.5, 0,   0.2, 0.2, 0.7, 0.5,  1,   0.3]
    TG = [0, 0, 0, 0.5, 1, 1, 0, 1, 0.7, 0.4, 0.7, 0.2, 0,   0.25, 0.5, 0.3]
    R = Img[:, :, 0].copy()
    G = Img[:, :, 1].copy()
    B = Img[:, :, 2].copy()
    for i in range(Label.max()+1):
        if i < len(TR):  # Load color from Table
            R[Label == i] = TR[i] * 255
            G[Label == i] = TG[i] * 255
            B[Label == i] = TB[i] * 255
        else:  # Generate random label color
            R[Label == i] = np.mod(i*i+4*i+5, 255)
            G[Label == i] = np.mod(i*10, 255)
            B[Label == i] = np.mod(i*i*i+7*i*i+3*i+30, 255)
    Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
    Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
    Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
    return Img


def save_seg_result(image, pred_mask, image_id=1):
    imcpy = image.copy()

    # save predict mask as PNG image
    mask_dir = os.path.join('result', 'predict_mask', str(image_id))
    os.makedirs(mask_dir, exist_ok=True)
    result_dir = os.path.join('result', 'segmentation', str(image_id))
    os.makedirs(result_dir, exist_ok=True)

    pred_mask = cv2.resize(pred_mask, image.shape[1::-1],interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(mask_dir, str(image_id)+'.png'), pred_mask)

    im = OverLayLabelOnImage(image, pred_mask).astype('uint8')
    cv2.imwrite(os.path.join(result_dir, str(image_id)+'.png'), im)

    ###################bbox##################
    pred_masks = [(pred_mask == v) for v in range(15)]
    for i in range(1, 15):
        channel = pred_masks[i].astype('uint8')
        contours, heirarchy = cv2.findContours(
            channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
        for idx, cnt in enumerate(contours):
            corners = find_corners_of_contour(cnt)
            cropped = crop_and_warp(imcpy, corners)
            cv2.imwrite(os.path.join(result_dir, TOTAL_CLASSES[i]+'_'+str(idx) + '.png'), cropped)

    cv2.imwrite(os.path.join(result_dir, str(image_id)+'bbox.png'), image)


def loadFrozenModel(frozen_model_name):
    with tf.io.gfile.GFile(frozen_model_name, 'rb') as f:
        restored_graph_def = tf.compat.v1.GraphDef()
        restored_graph_def.ParseFromString(f.read())
        if restored_graph_def is None:
          raise RuntimeError('Cannot find inference graph in tar archive.')
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def=restored_graph_def,name='')
        #input_tensor = graph.get_tensor_by_name('input_1:0')
        #conv_tensor = graph.get_tensor_by_name('output_layer/BiasAdd:0')
        #print(input_tensor)
        #print(conv_tensor)
    session = tf.compat.v1.Session(graph=graph)
    return session


def meanSubtract(image,mean_color= [103.939, 116.779, 123.68]):      
    image = image.astype('float32')
    image -= np.array(mean_color)
    return image


frozen_model_name = './pbonly/frozen_model.pb'
session = loadFrozenModel(frozen_model_name)

im_name = 'huk1'

image = cv2.imread(im_name+'.jpg')
imcpy=image.copy()
image = cv2.resize(image, (WIDTH, HEIGHT))

image = meanSubtract(image)

image_array = np.expand_dims(image, axis=0)

pr_mask = session.run('output_layer/BiasAdd:0',feed_dict={'input_1:0': image_array})#[image]

pr_mask = np.argmax(pr_mask, axis=-1)
pr_mask = pr_mask.squeeze().astype('uint8')


save_seg_result(imcpy, pr_mask, im_name)

