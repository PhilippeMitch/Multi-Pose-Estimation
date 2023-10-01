import time
import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torchvision.transforms  as T
from torchvision.models.detection import (keypointrcnn_resnet50_fpn, 
                                          KeypointRCNN_ResNet50_FPN_Weights)


class PoseEstimation:

    def __init__(self, args) -> None:
        self.args = args
        # create a model object from the keypointrcnn_resnet50_fpn class
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
        # call the eval() method to prepare the model for inference mode.
        self.model.eval()
        # preprocess the input image
        self.transform = T.Compose([T.ToTensor()])

        # create the list of keypoints.
        self.keypoints = [
            'nose', 'left_eye',
            'right_eye', 'left_ear',
            'right_ear', 'left_shoulder',
            'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist',
            'right_wrist', 'left_hip',
            'right_hip', 'left_knee', 
            'right_knee', 'left_ankle', 'right_ankle'
        ]

    # get the limbs
    def get_limbs_from_keypoints(self):
        limbs = [       
            [self.keypoints.index('right_eye'), self.keypoints.index('nose')],
            [self.keypoints.index('right_eye'), self.keypoints.index('right_ear')],
            [self.keypoints.index('left_eye'), self.keypoints.index('nose')],
            [self.keypoints.index('left_eye'), self.keypoints.index('left_ear')],
            [self.keypoints.index('right_shoulder'), self.keypoints.index('right_elbow')],
            [self.keypoints.index('right_elbow'), self.keypoints.index('right_wrist')],
            [self.keypoints.index('left_shoulder'), self.keypoints.index('left_elbow')],
            [self.keypoints.index('left_elbow'), self.keypoints.index('left_wrist')],
            [self.keypoints.index('right_hip'), self.keypoints.index('right_knee')],
            [self.keypoints.index('right_knee'), self.keypoints.index('right_ankle')],
            [self.keypoints.index('left_hip'), self.keypoints.index('left_knee')],
            [self.keypoints.index('left_knee'), self.keypoints.index('left_ankle')],
            [self.keypoints.index('right_shoulder'), self.keypoints.index('left_shoulder')],
            [self.keypoints.index('right_hip'), self.keypoints.index('left_hip')],
            [self.keypoints.index('right_shoulder'), self.keypoints.index('right_hip')],
            [self.keypoints.index('left_shoulder'), self.keypoints.index('left_hip')]
        ]
        return limbs

    def preprocessing(self, img):
        img_tensor = self.transform(img)
        # forward-pass the model
        # the input is a list, hence the output will also be a list
        output = self.model([img_tensor])[0]
        return output

    def draw_keypoints_per_person(self, img, all_keypoints, scores, 
                                  confs, keypoint_threshold=2, conf_threshold=0.9):
        edges = self.get_limbs_from_keypoints()
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
                
                # iterate for every limb 
                for limb_id in range(len(edges)):
                    # get different colors for the edges
                    rgb = colors.hsv_to_rgb([
                        limb_id/float(len(edges)), 1.0, 1.0
                    ])
                    rgb = rgb*255
                    # pick the start-point of the limb
                    limb_loc1 = keypoints[edges[limb_id][0], :2].detach().numpy().astype(np.int32)
                    # pick the start-point of the limb
                    limb_loc2 = keypoints[edges[limb_id][1], :2].detach().numpy().astype(np.int32)
                    # consider limb-confidence score as the minimum keypoint score among the two keypoint scores
                    limb_score = min(scores[person_id, edges[limb_id][0]], scores[person_id, edges[limb_id][1]])
                    # check if limb-score is greater than threshold
                    if limb_score> keypoint_threshold:
                        # draw the line for the limb
                        cv.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), tuple(rgb), 2, cv.LINE_AA)

        return img_copy

    def draw_keypoints(self, outputs, image):
        edges = self.get_limbs_from_keypoints()
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
    
    def _image_estimation(self):
        # Read the image using opencv 
        image = cv.imread(self.args.input)
        # To improve performance, optionally mark the image as immutable to
        # pass by reference.
        image.flags.writeable = False
        output = self.preprocessing(image)
        keypoints_img = self.draw_keypoints_per_person(image, output["keypoints"], 
            output["keypoints_scores"], output["scores"],keypoint_threshold=2)
        
        cv.imshow("Human pose estimation with RCNN", keypoints_img)
        # (this is necessary to avoid Python kernel form crashing)
        cv.waitKey(0)
        # Press q to close displayed window and stop the app
        if cv.waitKey(10) & 0xFF==ord('q'):
            # closing all open windows
            cv.destroyAllWindows()
        
    def _video_estimation(self):

        cap = cv.VideoCapture(self.args.input)

        while cap.isOpened():
            # Capture frame-by-frame
            success, image = cap.read()
            # if frame is read correctly ret is True
            if not success:
                print("Frame not found!!")
                break
            # To improve performance, optionally mark the image as immutable to
            # pass by reference.
            image.flags.writeable = False
            # Starting time for the process
            t1 = time.time()
            # Send this image to the model
            output = self.preprocessing(image)
            keypoints_img = self.draw_keypoints(output, image)
            # Ending time for the process
            t2 = time.time()
            # Number of frames that appears within a second
            fps = 1/(t2 - t1)
            # display the FPS
            cv.putText(image, 'FPS : {:.2f}'.format(fps), (int((image.shape[1] * 75) /100), 40), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (188, 205, 54), 2, cv.LINE_AA)
            cv.imshow("Human pose estimation with Kypoint RCNN", keypoints_img)
            if cv.waitKey(10) & 0XFF == ord('q'):
                break
        # Release everything if job is finished
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='Pose Estimation', description='Pose estimation with Keypoint RCNN')
    parser.add_argument('-i', '--input', type=str, default='../media/images/bus.jpg',
                        help='path to image/video file to use as input')
    args = parser.parse_args()
    pose_estimation = PoseEstimation(args)
    if args.input.endswith(('.jpg', '.jpeg', '.png')):
        pose_estimation._image_estimation()
    else:
        pose_estimation._video_estimation()
