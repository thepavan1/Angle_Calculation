import cv2
import numpy as np
import requests

url = "http://192.168.0.112:4747/shot.jpg"

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame is None:
        print("Failed to decode frame")
        continue
    cv2.imshow("Phone Cam Snapshot", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
