import cv2
import numpy as np
import numpy.linalg as LA
import collections
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util

'''
Convert normalized coordinates to absolute coordinates
'''
def convert_to_absolute(im_height, im_width, box):
    box_abs = []
    box_abs = [box[0] * im_height,
               box[1] * im_width,
               box[2] * im_height,
               box[3] * im_width]
    
    return box_abs


'''
Get hand area from face box
Hand area is the right area of the face box
'''
def get_hand_area(face_box, max_height, max_width):
    y_min = face_box[0]
    x_min = face_box[1]
    y_max = face_box[2]
    x_max = face_box[3]
    
    hand_area = [y_min, x_min-(x_max-x_min), y_max, x_min]

    box_width  = (hand_area[3]-hand_area[1])
    box_height = (hand_area[2]-hand_area[0])

    hand_area[0] -= (0.5*box_height)
    hand_area[1] -= (1.5*box_width)
    hand_area[2] += (1.0*box_height)
    
    if hand_area[0] <= 0.0:
        hand_area[0] = 0.0

    if hand_area[1] <= 0.0:
        hand_area[1] = 0.0

    if hand_area[2] >= max_height:
        hand_area[2] = max_height
        
    if hand_area[3] >= max_width:
        hand_area[3] = max_width

    return hand_area


'''
Hand sign classification
'''
def classify_hand_sign(model, hand_area):
    # Keras model(*.h5) style 
    input_img = cv2.resize(hand_area, (150, 150))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    
    input_img = np.expand_dims(input_img, axis=0)    
    input_img = input_img.astype(np.float32) / 255.
    
    preds = model(input_img)
    
    if np.amax(preds[0]) > 0.85:
        return np.argmax(preds[0])
    
#     # Tensorflow Saved Model style 
#     image = cv2.resize(image, (150, 150))
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     input_img = np.expand_dims(image, axis=0)    
#     input_img = input_img.astype(np.float32) / 255.
#     input_tensor = tf.constant(input_img)
    
#     result = model(input_tensor)
#     preds = result['dense_1'].numpy()


    
'''
Get all detected face boxes
'''
def get_all_face_box(model, image, max_boxes_to_detect):
    # 1. Face Detection
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    
    # 2. Get face boxes
    face_boxes = []
 
    im_height, im_width, _ = image.shape
    boxes=output_dict['detection_boxes']
    scores=output_dict['detection_scores']
    
    if not max_boxes_to_detect:
        max_boxes_to_detect = boxes.shape[0]
        
    for i in range(min(max_boxes_to_detect, boxes.shape[0])):
        if scores is None or scores[i] > 0.5:
            box_abs = convert_to_absolute(im_height, im_width, boxes[i])
            face_boxes.append(tuple(box_abs))

    return face_boxes


'''
Find face box to track
If the classification result of the hand area is 5, then track this face box
'''
def find_face_box_to_track(model, image, face_boxes):
    im_height, im_width, _ = image.shape
    
    for face_box in face_boxes:
        hand_area = get_hand_area(face_box, im_height, im_width)
        hand_area_img = image[int(hand_area[0]):int(hand_area[2]),
                              int(hand_area[1]):int(hand_area[3])]
        
        # invalid box check
        if hand_area_img.size != 0:
            label = classify_hand_sign(model, np.array(hand_area_img))
            if label == 5:
                return face_box


'''
Face box tracking using Euclidean distance
'''
def track_face_box(target_box, face_boxes):
    distances = LA.norm(np.array(target_box)-np.array(face_boxes), axis=1)
    if np.min(distances) < 50.0:
        return face_boxes[np.argmin(distances)]
                         

'''
Get command sign
'''
def get_command(model, image, face_box):
    im_height, im_width, _ = image.shape
    
    hand_area = get_hand_area(face_box, im_height, im_width)
    hand_area_img = image[int(hand_area[0]):int(hand_area[2]),
                          int(hand_area[1]):int(hand_area[3])]

    # invalid box check
    if hand_area_img.size != 0:
        label = classify_hand_sign(model, np.array(hand_area_img))
        if label is not None:
            return (hand_area, label)
        
        
'''
Visualize box and string on background image
'''
def visualize_box(bg_image, box, display_str, color):
    ymin, xmin, ymax, xmax = box
    vis_util.draw_bounding_box_on_image_array(bg_image,
                                              ymin,
                                              xmin,
                                              ymax,
                                              xmax,
                                              color=color,
                                              thickness=4,
                                              display_str_list=[display_str],
                                              use_normalized_coordinates=False)
