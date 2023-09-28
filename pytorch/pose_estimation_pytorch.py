import time
import cv2 as cv
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torchvision.transforms  as T


cap = cv.VideoCapture("https://assets.mixkit.co/videos/preview/mixkit-two-couples-walking-near-a-cabin-in-the-woods-42730-large.mp4")
# cap = cv.VideoCapture("../../Media/dance.mp4")

# create a model object from the keypointrcnn_resnet50_fpn class
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# call the eval() method to prepare the model for inference mode.
model.eval()
# preprocess the input image
transform = T.Compose([T.ToTensor()])

# create the list of keypoints.
keypoints = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee', 
    'right_knee', 
    'left_ankle',
    'right_ankle'
]

# get the limbs
def get_limbs_from_keypoints(keypoints):
    limbs = [       
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
        ]
    return limbs


edges = get_limbs_from_keypoints(keypoints)

def preprocessing(img):
    img_tensor = transform(img)
    # forward-pass the model
    # the input is a list, hence the output will also be a list
    output = model([img_tensor])[0]
    return output


def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # pick a set of N color-ids from the spectrum
    color_id = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
    # iterate for every person detected
    for person_id in range(len(all_keypoints)):
      # check the confidence score of the detected person
      if confs[person_id]>conf_threshold:
        # grab the keypoint-locations for the detected person
        keypoints = all_keypoints[person_id, ...]
        # grab the keypoint-scores for the keypoints
        scores = all_scores[person_id, ...]
        # iterate for every keypoint-score
        for kp in range(len(scores)):
            # check the confidence score of detected keypoint
            if scores[kp]>keypoint_threshold:
                # convert the keypoint float-array to a python-list of intergers
                keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                # pick the color at the specific color-id
                color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                # draw a cirle over the keypoint location
                cv.circle(img_copy, keypoint, 2, color, -1)

    return img_copy

def draw_keypoints(outputs, image):
    # the `outputs` is list which in-turn contains the dictionaries
    for i in range(len(outputs['keypoints'])):
        keypoints = outputs['keypoints'][i].cpu().detach().numpy()
        # proceed to draw the lines if the confidence score is above 0.9
        if outputs['scores'][i] > 0.9:
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                            3, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                # join the keypoint pairs to draw the skeletal structure
                cv.line(
                    image, 
                    (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                    (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])), 
                    tuple(rgb), 2, lineType=cv.LINE_AA
                )
        else:
            continue
    return image

while cap.isOpened():
    # Capture frame-by-frame
    success, image = cap.read()
    # if frame is read correctly ret is True
    if not success:
        print("Frame not found!!")
        break
    # To improve performance, optionally mark the image as immutable to
    #   pass by reference.
    image.flags.writeable = False
    # Starting time for the process
    t1 = time.time()
    # Send this image to the model
    output = preprocessing(image)
    keypoints_img = draw_keypoints(output, image)
    #keypoints_img = draw_keypoints_per_person(image, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)
    # Ending time for the process
    t2 = time.time()
    # Number of frames that appears within a second
    fps = 1/(t2 - t1)
    # display the FPS
    cv.putText(image, 'FPS : {:.2f}'.format(fps), (int((image.shape[1] * 75) /100), 40), cv.FONT_HERSHEY_SIMPLEX, 1, (188, 205, 54), 2, cv.LINE_AA)
    cv.imshow("Human pose estimation with RCNN", keypoints_img)
    if cv.waitKey(10) & 0XFF == ord('q'):
        break
# Release everything if job is finished
cap.release()
cv.destroyAllWindows()