import cv2
import numpy as np

net = cv2.dnn.readNet("C:/Users/ASUS/OneDrive/Desktop/helmet_detection_correct2/Helmet-Detection-Model-main/HelmetDetection/Helmet.cfg",
                      "C:/Users/ASUS/OneDrive/Desktop/helmet_detection_correct2/Helmet-Detection-Model-main/HelmetDetection/Helmet.weights")

layer_names = net.getLayerNames()
out_layer_indexes = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in out_layer_indexes.reshape(-1)]

with open("C:/Users/ASUS/OneDrive/Desktop/helmet_detection_correct2/Helmet-Detection-Model-main/HelmetDetection/Helmet.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

conf_threshold = 0.3
nms_threshold = 0.3

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            if scores.size > 0:
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    print(f'Detections: {len(indexes)}')

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = [0, 255, 0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
