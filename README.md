# Multi Pose Estimation
Multi-person pose estimation (aka, keypoint detection) is one of the fundamental computer vision tasks and has a wide range of applications such as action recognition, augmented reality, human computer interaction, pedestrian tracking and re-identification, etc.

### Multi Pose Estimation with Yolov8
YOLOv8 pretrained Pose models, Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

### Multi Pose Estimation with PyTorch Keypoint RCNN
PyTorch provides a pre-trained Keypoint RCNN model with ResNet50 base which has been trained on the COCO keypoint dataset. Keypoint RCNN is an algorithm for finding keypoints on images containing a human.