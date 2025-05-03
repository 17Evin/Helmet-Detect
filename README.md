# Helmet Detection Using YOLOv3

This project demonstrates helmet detection in images and videos using the YOLOv3 object detection model. The model is pre-trained on the custom dataset containing helmet objects.

## Features

- **Real-time helmet detection** in images and videos.
- Utilizes **YOLOv3** for fast and efficient object detection.
- Capable of detecting helmets in both **image files** and **video streams** (including webcam).
- Adjustable confidence threshold for detection accuracy.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- OpenCV
- Numpy

### Install dependencies:

You can install the required Python dependencies using the following command:

```bash
pip install -r requirements.txt
````

## Setup

1. **Clone the repository:**

   Clone this repository to your local machine using:

   ```bash
   git clone https://github.com/your-username/helmet-detection-yolov3.git
   cd helmet-detection-yolov3
   ```

2. **Download YOLOv3 files:**

   * Download the **YOLOv3 configuration file** (`yolov3-helmet.cfg`) and **YOLOv3 weights** (`yolov3-helmet.weights`) from a trusted source.

   * Create a `yolo-helmet/` folder in the project directory and place these files inside it.

   * **Download class names file** (`helmet.names`) and place it in the `yolo-helmet/` folder.

3. **Prepare input images or videos:**

   * Place your input image (`img.jpg`) in the project directory.
   * Optionally, place your video file (`video.mp4`) in the project directory for video input detection.

## Usage

### For Image Input

1. Set `input_type = "image"` in the script (`helmet_detection.py`).

2. Run the script:

   ```bash
   python helmet_detection.py
   ```

   The script will load the image, perform helmet detection, and display the output with bounding boxes around the detected helmets.

### For Video Input

1. Set `input_type = "video"` in the script (`helmet_detection.py`).

2. Change the video input to `video.mp4` (or use `"0"` for webcam).

3. Run the script:

   ```bash
   python helmet_detection.py
   ```

   The script will process the video or webcam stream and display real-time helmet detection.

### Optional: Test with Webcam

To use your webcam instead of a video file, change the input to `cap = cv2.VideoCapture(0)` in the script. Then run:

```bash
python helmet_detection.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* **YOLOv3**: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
* **OpenCV**: [https://opencv.org/](https://opencv.org/)
* Thanks to all the contributors and resources that helped make this project possible.


Just copy and paste this into your `README.md` file, and you're good to go! Let me know if you need further adjustments.
```
