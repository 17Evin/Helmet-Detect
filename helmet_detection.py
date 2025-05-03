import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNetFromDarknet("yolo-helmet/yolov3-helmet.cfg", "yolo-helmet/yolov3-helmet.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("yolo-helmet/helmet.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Choose input type (image or video)
input_type = "video"  # Change to "video" for video input

if input_type == "image":
    # Load an image
    image = cv2.imread("img.jpg")
    height, width, channels = image.shape

    # Create a blob and perform a forward pass
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.3:  # Helmet class with threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Helmet Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif input_type == "video":
    # Open video capture (use 0 for webcam or provide a file path)
    cap = cv2.VideoCapture("video.mp4")  # Change to 0 for webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Create a blob and perform a forward pass
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == 0 and confidence > 0.3:  # Helmet class with threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Draw bounding box and label
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Helmet Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
