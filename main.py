# Main.py
from ultralytics import YOLO
import cv2
import argparse
import numpy as np
from utils import draw_boxes_responsive
import tkinter as tk
from tkinter import filedialog, messagebox

class InteractiveDisplay:
    def __init__(self, window_name='YOLOv8 Detection'):
        self.window_name = window_name
        self.scale = 1.0
        self.min_scale = 0.2
        self.max_scale = 5.0
        self.zoom_speed = 0.1
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None
        self.dragging = False
        self.original_frame = None
        self.results = None
        self.is_fullscreen = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_events)

    def mouse_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = self.zoom_speed if flags > 0 else -self.zoom_speed
            self.scale = max(self.min_scale, min(self.max_scale, self.scale + delta))
            self.limit_offsets()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (x, y)
            self.limit_offsets()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.toggle_fullscreen()

    def limit_offsets(self, img_shape=None, window_size=None):
        if img_shape is None and self.original_frame is not None:
            img_shape = self.original_frame.shape[:2]
        if window_size is None:
            window_size = (800, 600)
        if img_shape:
            h, w = img_shape
            scaled_w, scaled_h = int(w * self.scale), int(h * self.scale)
            win_w, win_h = window_size
            self.offset_x = min(0, max(self.offset_x, win_w - scaled_w))
            self.offset_y = min(0, max(self.offset_y, win_h - scaled_h))

    def set_initial_view(self, window_size=None):
        if self.original_frame is None:
            return
        h, w = self.original_frame.shape[:2]
        if window_size is None:
            window_size = (800, 600)
        win_w, win_h = window_size
        self.scale = min(win_w / w, win_h / h)
        scaled_w = int(w * self.scale)
        scaled_h = int(h * self.scale)
        self.offset_x = (win_w - scaled_w) // 2
        self.offset_y = (win_h - scaled_h) // 2
        self.limit_offsets(img_shape=(h, w), window_size=window_size)

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        try:
            win_rect = cv2.getWindowImageRect(self.window_name)
            win_w, win_h = win_rect[2], win_rect[3]
        except:
            win_w, win_h = 1920, 1080
        self.set_initial_view(window_size=(win_w, win_h))

    def display_frame(self):
        if self.original_frame is None or self.results is None:
            return None
        if self.is_fullscreen:
            try:
                win_rect = cv2.getWindowImageRect(self.window_name)
                win_w, win_h = win_rect[2], win_rect[3]
            except:
                win_w, win_h = 1920, 1080
        else:
            win_w, win_h = 800, 600
        scaled_h, scaled_w = int(self.original_frame.shape[0] * self.scale), int(self.original_frame.shape[1] * self.scale)
        scaled_frame = cv2.resize(self.original_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        annotated = draw_boxes_responsive(scaled_frame, self.results, self.scale)
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x1_c = max(self.offset_x, 0)
        y1_c = max(self.offset_y, 0)
        x1_i = max(-self.offset_x, 0)
        y1_i = max(-self.offset_y, 0)
        vis_w = min(annotated.shape[1] - x1_i, win_w - x1_c)
        vis_h = min(annotated.shape[0] - y1_i, win_h - y1_c)
        canvas[y1_c:y1_c + vis_h, x1_c:x1_c + vis_w] = annotated[y1_i:y1_i + vis_h, x1_i:x1_i + vis_w]
        cv2.imshow(self.window_name, canvas)
        return canvas

class DetectionGUI:
    def __init__(self, model_path='yolov8n.pt', conf_thresh=0.5):
        self.root = tk.Tk()
        self.root.title("YOLOv8 Object Detection GUI")
        self.root.geometry("600x400")  # Balanced initial size
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.image_path = None

        # GUI Elements
        self.label = tk.Label(self.root, text="Welcome to YOLOv8 Detection", font=("Arial", 14))
        self.label.pack(pady=20)

        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.detect_button = tk.Button(self.root, text="Run Detection", command=self.run_detection, state=tk.DISABLED)
        self.detect_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.status_label.pack(pady=10)

        self.root.mainloop()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if self.image_path:
            self.status_label.config(text=f"Image loaded: {self.image_path}")
            self.detect_button.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="No image selected")

    def run_detection(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image loaded")
            return

        img = cv2.imread(self.image_path)
        if img is None:
            messagebox.showerror("Error", f"Unable to read image from {self.image_path}")
            return

        model = YOLO(self.model_path)
        results = model(img, conf=self.conf_thresh)

        display = InteractiveDisplay()
        display.original_frame = img
        display.results = results
        win_w, win_h = 800, 600
        display.set_initial_view(window_size=(win_w, win_h))

        while True:
            saved_canvas = display.display_frame()
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s') and saved_canvas is not None:
                cv2.imwrite('output_image.jpg', saved_canvas)
            elif key == ord('r'):
                display.set_initial_view()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse arguments for model and conf (optional, can be hardcoded if preferred)
    parser = argparse.ArgumentParser(description='YOLOv8 Detection GUI')
    parser.add_argument('--model', type=str, default='yolov8x.pt', help='YOLOv8 model path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()

    DetectionGUI(model_path=args.model, conf_thresh=args.conf)
