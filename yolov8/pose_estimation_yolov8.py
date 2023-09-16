from ultralytics import YOLO
import time
import argparse
import cv2


class PoseEstimation:

    def __init__(self, args) -> None:
        self.args = args

    def _image_estimation(self, model):
        # Predict with the model
        results = model(self.args.input)
        keypoints = results[0].keypoints
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        # Press q to close displayed window and stop the app
        if cv2.waitKey(10) & 0xFF==ord('q'):
            # closing all open windows
            cv2.destroyAllWindows()

    def _video_estimation(self, model):
        cap = cv2.VideoCapture(args.input)
        # https://assets.mixkit.co/videos/preview/mixkit-two-couples-walking-near-a-cabin-in-the-woods-42730-large.mp4
        # https://player.vimeo.com/external/376858866.sd.mp4?s=d963a730f18ac7e9e93a0664dec43eab3ae41617&amp;profile_id=164&amp;oauth2_token_id=57447761
        while cap.isOpened():
            # Read the frame from the video
            res, frame = cap.read()

            if res:
                # Starting time for the process
                t1 = time.time()
                # Predict with the model
                results = model(frame)
                # Ending time for the process
                t2 = time.time()
                # Number of frames that appears within a second
                fps = 1/(t2 - t1)
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                dim = (frame.shape[1], frame.shape[0])
                resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                # display the FPS
                cv2.putText(annotated_frame, 'FPS : {:.2f}'.format(fps), (int((frame.shape[1] * 75) /100), 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (188, 205, 54), 2, cv2.LINE_AA)
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
        
        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    # load an official model
    parser = argparse.ArgumentParser(prog='Pose Estimation', description='Pose estimation with yolov8')
    parser.add_argument('-i', '--input', type=str, default='bus.jpg',
                        help="path to video/image file to use as input")

    args = parser.parse_args()
    pose_estimation = PoseEstimation(args)

    model = YOLO('yolov8n-pose.pt')

    if args.input.endswith(('.jpg', '.jpeg', '.png')):
        pose_estimation._image_estimation(model)
    else:
        pose_estimation._video_estimation(model)