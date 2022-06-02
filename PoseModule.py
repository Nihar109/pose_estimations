import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, staticMode=False,
                 model_complexity=1,
                 smooth=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 minDetectionCon=0.5,
                 minTrackCon=0.5):

        self.staticMode = staticMode
        self.modelComplexity = model_complexity
        self.smooth = smooth
        self.enableSegmentation = enable_segmentation
        self.smoothSegmentation = smooth_segmentation
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.staticMode, model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smooth, enable_segmentation=self.enableSegmentation,
                                     smooth_segmentation=self.smoothSegmentation,
                                     min_detection_confidence=self.minDetectionCon,
                                     min_tracking_confidence=self.minTrackCon)
        self.drawLandmarkSpec = self.mpDraw.DrawingSpec(
            thickness=5, circle_radius=2, color=(255,0,0))
        self.drawConnectionSpec = self.mpDraw.DrawingSpec(
            thickness=2, color=(0,255,0))
    def findPose(self, img, draw=True):

        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRBG)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                      self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, o = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w) , int(lm.y *h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255,0,0),cv2.FILLED)
        return lmlist


def main():
    cap = cv2.VideoCapture('video1.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.findPosition(img, draw=False)
        print(lmlist)
        if lmlist is not None:
            cv2.circle(img, (lmlist[13][1], lmlist[13][2]), 10, (255,0,0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

