import cv2
import os
from threading import Thread
import subprocess

# if True, pull from webcam; if False, run OpenNI with ROS (used for PrimeSense)
WEBCAM = True
DISP_SCALE = 0.7

class WebcamImageGetter:
    def __init__(self):
        self.currentFrame = None
        self.capture = cv2.VideoCapture(0)
        self.keep_going = True

    def start(self):
        Thread(target=self.updateFrame).start()

    def updateFrame(self):
        while self.keep_going:
            ret, frame = self.capture.read()
            if ret:
                self.currentFrame = cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)

    def getFrame(self):
        return self.currentFrame

    def end(self):
        pass

class OpenNIImageGetter:
    def __init__(self):
        self.currentFrame = None
        self.keep_going = True

    def start(self):
        Thread(target=self.listener).start()
        self.p = subprocess.Popen(["roslaunch", "openni2_launch", "openni2.launch"])

    def callback(self, data, bridge):
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
        self.currentFrame = cv2.resize(frame, dsize=(0, 0), fx=DISP_SCALE, fy=DISP_SCALE)

    def listener(self):
        bridge = CvBridge()
        rospy.init_node("listener", anonymous=True, disable_signals=True)
        rospy.Subscriber("/camera/rgb/image", Image, self.callback, bridge)
        while self.keep_going:
            pass

    def getFrame(self):
        return self.currentFrame

    def end(self):
        self.p.terminate()

if __name__ == "__main__":
    print "Press enter to quit."
    if WEBCAM:
        w = WebcamImageGetter()
    else:
        import rospy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        w = OpenNIImageGetter()
    w.start()
    while True:
        frame = w.getFrame()
        if frame is None:
            continue
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == 10:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            w.keep_going = False
            w.end()
            break
