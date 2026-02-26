import cv2

# Function draws bounding boxes and labels scaled responsively to zoom
# This function receives the scaled image and original detection results,
# it adjusts bounding boxes to the current scaled image size for precise apposition.
def draw_boxes_responsive(img, results, scale_factor=1.0):
    annotated_img = img.copy()
    base_font = 0.5
    base_thick = 2
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # box coordinates in original image (float)
            x1o, y1o, x2o, y2o = map(float, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())

            # Adjust bounding box with margin (tighter)
            margin = int(3 * (1 - conf))  # smaller margin for high confidence
            x1o = max(0, x1o + margin)
            y1o = max(0, y1o + margin)
            x2o = min(img.shape[1]/scale_factor, x2o - margin)
            y2o = min(img.shape[0]/scale_factor, y2o - margin)

            # Scale to displayed image size
            x1 = int(x1o * scale_factor)
            y1 = int(y1o * scale_factor)
            x2 = int(x2o * scale_factor)
            y2 = int(y2o * scale_factor)

            label = f"{result.names[cls_id]} {conf:.2f}"

            # Scale bounding box thickness and font size with zoom
            font_scale = max(0.3, base_font * scale_factor)
            thickness = max(1, int(base_thick * scale_factor))

            # Calculate text size for background
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y1 = max(0, y1 - h - baseline - 10)

            # Draw filled rectangle with transparency
            cv2.rectangle(annotated_img, (x1, label_y1), (x1 + w + 10, y1), (0, 0, 0), -1)
            overlay = annotated_img.copy()
            cv2.rectangle(overlay, (x1, label_y1), (x1 + w + 10, y1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_img, 0.3, 0, annotated_img)

            # Draw label text with anti aliasing and outline
            cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    return annotated_img
