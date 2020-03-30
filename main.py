from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

threshold_val = 155
min_area = 100

# vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

data = {}

while True:
    
    # grab the frame from the threaded video stream
    frame = vs.read()

    img2gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, imge_bin = cv2.threshold(img2gray, threshold_val, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(imge_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours_filtered = []

    for cnt in contours:
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        
        if M['m00'] != 0 and area > min_area:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            dist = [cv2.pointPolygonTest(cnt2,(cx,cy),True) for cnt2 in contours_filtered]
            total = [True if x < 0 else False for x in dist]
            if dist == [] or True in total:
                contours_filtered.append(cnt)
   
    for cnt in contours_filtered:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            area = cv2.contourArea(cnt)

            x,y,w,h = cv2.boundingRect(cnt)
            cv2.circle(frame, (cx,cy),2, (255, 0, 0))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    fps.update()

    cv2.imshow("Frame", frame)
    cv2.imshow("Frame binary", imge_bin)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()