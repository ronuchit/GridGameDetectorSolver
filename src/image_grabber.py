import cv2
from threading import Thread
import subprocess

class WebcamImageGetter:
    def __init__(self, disp_scale):
        self.currentFrame = None
        self.capture = cv2.VideoCapture(0)
        self.keep_going = True
        self.disp_scale = disp_scale

    def start(self):
        Thread(target=self.updateFrame).start()

    def updateFrame(self):
        while self.keep_going:
            ret, frame = self.capture.read()
            if ret:
                self.currentFrame = cv2.resize(frame, dsize=(0, 0), fx=self.disp_scale, fy=self.disp_scale)

    def getFrame(self):
        return self.currentFrame

    def end(self):
        pass

class OpenNIImageGetter:
    def __init__(self, disp_scale):
        self.currentFrame = None
        self.keep_going = True
        self.disp_scale = disp_scale

    def start(self):
        Thread(target=self.listener).start()
        self.p = subprocess.Popen(["roslaunch", "openni2_launch", "openni2.launch"])

    def callback(self, data, bridge):
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
        self.currentFrame = cv2.resize(frame, dsize=(0, 0), fx=self.disp_scale, fy=self.disp_scale)

    def listener(self):
        import rospy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        bridge = CvBridge()
        rospy.init_node("listener", anonymous=True, disable_signals=True)
        rospy.Subscriber("/camera/rgb/image", Image, self.callback, bridge)
        while self.keep_going:
            pass

    def getFrame(self):
        return self.currentFrame

    def end(self):
        self.p.terminate()
