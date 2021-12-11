import cv2
import numpy as np

def main(IMAGE_PATH):
    # yolo load
    net = cv2.dnn.readNet("yolov4-obj_2_best.weights", "yolov4-obj_2.cfg")
    classes = []

    with open('obj.names', "r", encoding='UTF-8') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    # print("-"*30)
    # print(layer_names[326])
    # print(layer_names[352])
    # print(layer_names[378])
    # print("-"*30)
    # print(net.getUnconnectedOutLayers())
    # print("-"*30)
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img = cv2.imread(IMAGE_PATH)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # show infomation
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # x, y coordinate
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # noise reduction
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    # show images
    try:
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 3), font, 1, color, 3)

                print(x, y, w, h)
                crop_image = img[y : y+h, x : x+w]
                # cv2.imshow("crop",crop_image)
                cv2.imwrite(f'./result_images/{i:02d}_{label}_{IMAGE_PATH}', crop_image)
                
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("cannot show image")

    # crop_image = img[x : x+w, y : y+h]
    # cv2.imshow("crop",crop_image)
    cv2.imwrite(f'./pred_images/{IMAGE_PATH}', img)


if __name__ in "__main__":
    TEST_IMG_PATH = 'wakuwaku.jpg'
    main(TEST_IMG_PATH)
