import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import matplotlib.pyplot as plt
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import math
import requests
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet



flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


#pts = [deque(maxlen=9999) for _ in range(9999)]


def main(_argv):
    
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    dtime =dict()
    dwell_time=dict()
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        #background_sub = cv2.BackgroundSubtractorMOG2()
    frame_num = 0
    # while video is running
    directionPoints = 10
    direction = "" 
    pts = [deque(maxlen=9999) for _ in range(9999)]
    #pts = deque()
    counter =0
    (dx,dy) = (0,0)
    
    
  

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        
        
        
        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        
        

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            i += 1
            #bbox_center_point(x,y)
            Cx =int(((bbox[0])+(bbox[2]))/2) 
            Cy =int(((bbox[1])+(bbox[3]))/2)
            center = (Cx,Cy)
            #track_id[center]
            angle = 0

            pts[track.track_id].append(center)
            

            thickness = 2
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)
            
			# draw motion path
            
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j]is None:
                    #pts[track.track_id].append[track_id]
                    continue
                    #starting_point=center
                    #ending_point = center
                    #cv2.line(frame,starting_point,ending_point,(color),thickness)
                else:
                    for pt in range(len(pts[track.track_id])):
                        if not pt+1==1:
                            starting_point =pts[track.track_id][pt][0],pts[track.track_id][pt][1]
                            ending_point = pts[track.track_id][pt-1][0],pts[track.track_id][pt-1][1]
                            cv2.line(frame, starting_point, ending_point, color, thickness)
                            
            pts[track.track_id].appendleft(center)                
            for i in np.arange(1,len(pts[track.track_id])):
                j = len(pts[track.track_id]) - i
            
                if (len(pts[track.track_id])>directionPoints):
                        
                    
                    if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                        
                        continue
                    elif pts[-directionPoints] is not None and not j+1==1:
                        x1 = pts[track.track_id][-directionPoints][0]
                        x2 = pts[track.track_id][j][0]
                            
                        y1 =   pts[track.track_id][j][1]  
                        y2 = pts[track.track_id][-directionPoints][1]
                        
                        dx = pts[track.track_id][j][0] - pts[track.track_id][-directionPoints][0] #(x2-x1)
                        dy = pts[track.track_id][-directionPoints][1] - pts[track.track_id][j][1] #(y2-y1)                
                        (dirX,dirY) = ("","")

                        angle = math.atan2(y2-y1, x2-x1) # tangent of two points 
                        angle = math.degrees(angle) #converting radian into degree
                        angle = round(angle)    
                        
                        if angle >= 150 or angle <= -150:
                            #if (dx) < -15:
                            dirX = "Left"
                            
                        if angle <= 30 or angle >= -30:
                        #    if (dx) > 15:
                            dirX = "Right"
                        
                        if angle >= 60 or angle <= 120:
                        #    if (dy) > 15:
                            dirX = "Up"
                        
                        if angle >= -120 or angle <= -60:
                        #    if (dy) < -15 and (dx) < -15:
                            dirX = "Down"
                        
                        if angle > 30 or angle < 60:
                        #    if (dy) > 15 and (dx) > 15:
                            dirX = "Up-Right"
                        
                        if angle > 120 or angle < 150:
                        #    if (dy) > 15 and (dx) < -15:
                            dirX = "Up-Left"
                        
                        if angle > -150 or angle < -120:
                        #    if (dy) < -15 and (dx) < -15:
                            dirX = "Down-Left"
                        
                        if angle > -60 or angle < -30:
                        #    if (dy) < -15 and (dx) > 15:
                            dirX = "Down-Right"
                            
                        if dirX != "" and dirY != "":
                           direction = "{}-{}".format(dirY,dirX)
                        else:
                           direction = dirX if dirX != "" else dirY
            
            if direction  == "Right":
                    track.Rcnts  += 1
            else:
                if track.Rcnts >=2:
                    track.Rtotal += 1
                track.Rcnts =0
            
            
            if direction  == "Left":
                track.Lcnts +=1
            else:
                if track.Lcnts >=2:
                    track.Ltotal +=1
                track.Lcnts =0

            if direction == "Up":
                track.Ucnts +=1
            else:
                if track.Ucnts >=2:
                    track.Utotal +=1
                track.Ucnts =0


            if direction == "Down":
                track.Dcnts +=1
            else:
                if track.Dcnts >=2:
                    track.Dtotal +=1
                track.Dcnts =0
                
            if direction == "Up-Right":
                track.UpRight_cnts +=1
            else:
                if track.UpRight_cnts >=2:
                    track.UpRight_total +=1
                track.UpRight_cnts =0
                
            if direction == "Up-Left": #South-West
                track.UpLeft_cnts +=1
            else:    
                if track.UpLeft_cnts >=2:
                    track.UpLeft_total += 1
                track.UpLeft_cnts =0
                
            if direction == "Down-Right":#"North-East":
                track.DownRight_cnts +=1
            else:
                if track.DownRight_cnts >=2:
                    track.DownRight_total += 1
                track.DownRight_cnts =0
                
                
            if direction == "Down-Left":#"North-West":
                track.DownLeft_cnts += 1
            else:
                if track.DownLeft_cnts >=2:
                    track.DownLeft_total +=1
                track.DownLeft_cnts =0
                
            print("Person ID: {}, Right cnts: {},  Left cnts: {}, Up cnts: {}, Down cnts : {}, Up-Right cnts: {}, Up-Left cnts : {}, Down-Right cnts: {}, Down-Left cnts: {}".format(str(track.track_id),str(int(track.Rtotal)),
            str(int(track.Ltotal)),str(int(track.Utotal)),str(int(track.Dtotal)),str(int(track.UpRight_total)),str(int(track.UpLeft_total)),str(int(track.DownRight_total)),str(int(track.DownLeft_total))))                  
        
            
            
                        
                        
            cv2.putText(frame, direction, (int(bbox[2]-10), int(bbox[3]-10)), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 1)
        
            cv2.putText(frame, "dx: {}, dy: {}".format(dx, dy),(int(bbox[2]+10), int(bbox[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            cv2.putText(frame, "Angle: {}".format(angle),(int(bbox[2]+30), int(bbox[3]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        

                    

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


# Convert darknet weights to tensorflow model
#python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video
#python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

# Run yolov4 deep sort object tracker on webcam (set video flag to 0)
#python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4