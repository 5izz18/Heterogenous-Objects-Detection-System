YOLOv8 Interactive Object Detection GUI

This project is a Python-based interactive object detection system built using YOLOv8 and OpenCV with a graphical interface using Tkinter. It allows users to load an image, run object detection, and view results with interactive zoom, pan, and responsive bounding boxes.
The system improves visualization by dynamically scaling bounding boxes and labels based on zoom level, making detection clearer for high-resolution images.

Features:
Image-based object detection using Ultralytics YOLOv8
GUI interface using Tkinter
Interactive zoom using mouse scroll
Drag-to-pan image navigation
Responsive bounding boxes with confidence scores
Fullscreen toggle support
Save detected output image
Adjustable confidence threshold

Technologies Used:
Python
OpenCV
NumPy
Tkinter
YOLOv8 (Ultralytics)

Project Structure:
├── main.py          # GUI and detection pipeline
├── utils.py         # Bounding box rendering utilities
├── yolov8n.pt       # Pretrained YOLOv8 model (download separately)

How It Works:
User selects an image using the GUI.
The YOLOv8 model detects objects in the image.
Bounding boxes and labels are drawn dynamically.
Users can zoom and pan to inspect detections interactively.

Installation:
Step 1: Clone Repository
git clone https://github.com/5izz18/yolov8-detection-gui.git
cd yolov8-detection-gui
Step 2: Install Dependencies
pip install ultralytics opencv-python numpy
Step 3: Download YOLOv8 Model

Download pretrained weights from the official Ultralytics repository.

Run the Project
python main.py

Controls
Mouse Scroll → Zoom
Left Drag → Pan
Right Click → Toggle Fullscreen
Press S → Save Output
Press Q → Exit

Future Improvements:

Real-time webcam detection
Custom dataset training
Detection statistics dashboard
Export results to CSV
![output_image](https://github.com/user-attachments/assets/54eb59d6-fd99-4529-a665-4e38f6e623e5)

