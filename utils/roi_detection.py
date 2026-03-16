"""
ROI (Region of Interest) detection for traffic signs
Uses color-based segmentation and contour detection

How it works:
 1. Convert the frame to HSV color space
 2. Find red, blue, and yellow colored areas (these are sign colors)
 3. Find the shape outlines (contours) of those colored areas
 4. Filter out shapes that are too small, too large, or the wrong shape
 5. Return bounding boxes around likely sign locations
"""
import cv2
import numpy as np
from config import (
    HSV_RED_LOWER1, HSV_RED_UPPER1, HSV_RED_LOWER2, HSV_RED_UPPER2,
    HSV_BLUE_LOWER, HSV_BLUE_UPPER, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER,
    MIN_CONTOUR_AREA, MAX_CONTOUR_AREA,
    CONFIDENCE_THRESHOLD
)


def detect_color_regions(frame):
    """
    Finds all red, blue, and yellow areas in the image.
    
    We use HSV color space instead of RGB because HSV is more consistent
    across different lighting conditions — a red sign in shadow still looks
    red in HSV, but might look brown or dark in RGB.
    
    Returns a binary mask (black/white image) where white = possible sign area.
    """
    # Convert the image from BGR (default in OpenCV) to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red is tricky in HSV — it wraps around from 180 back to 0
    # So we need two ranges to catch all shades of red
    red_mask1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)   # Low-hue reds
    red_mask2 = cv2.inRange(hsv, HSV_RED_LOWER2, HSV_RED_UPPER2)   # High-hue reds
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)                  # Combine both

    # Simple single-range detection for blue and yellow signs
    blue_mask = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    yellow_mask = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)

    # Merge all three color masks into one — any sign color counts
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, blue_mask), yellow_mask)

    # Clean up the mask using morphological operations:
    # MORPH_CLOSE fills small holes inside detected regions (e.g. white text on red sign)
    # MORPH_OPEN removes small random dots (noise from similar-colored objects in background)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask


def extract_contours(mask):
    """
    Finds the outlines (contours) of all white regions in the binary mask.
    Each contour is the boundary of a colored region — possibly a sign.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours_by_shape(contours, frame_shape):
    """
    Filters detected contours to only keep shapes that could realistically be a sign.
    
    We remove:
    - Very small regions (noise/artifacts)
    - Very large regions (sky, road, buildings)
    - Very thin or very wide shapes (not traffic signs — they're roughly square)
    - Very sparse shapes (too few pixels relative to the bounding box)
    
    Returns a list of bounding boxes for the shapes that pass all filters.
    """
    bounding_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip if too small (just noise) or too large (background objects)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        # Get the bounding rectangle around this contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check aspect ratio — signs are roughly square or slightly rectangular
        # Very thin or very wide shapes are probably not signs
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

        # Check extent — how densely the contour fills its bounding box
        # A very sparse contour (e.g. just an outline) is probably not a sign
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        if extent < 0.3:  # Less than 30% filled = too sparse
            continue

        bounding_boxes.append((x, y, w, h))

    return bounding_boxes


def make_square_bbox(x, y, w, h, frame_shape):
    """
    Converts a rectangular bounding box into a square.
    
    The CNN model expects square inputs (64x64), so we expand the shorter
    side of the detection box to make it square. We center it on the
    original detection and make sure it stays within the image boundaries.
    """
    # Use the longer side as the square size
    size = max(w, h)

    # Center the square on the original detection
    center_x = x + w // 2
    center_y = y + h // 2
    new_x = max(0, center_x - size // 2)
    new_y = max(0, center_y - size // 2)

    # Make sure the box doesn't go outside the image
    if new_x + size > frame_shape[1]:
        new_x = frame_shape[1] - size
    if new_y + size > frame_shape[0]:
        new_y = frame_shape[0] - size

    new_x = max(0, new_x)
    new_y = max(0, new_y)

    # Clamp size in case it still overflows at the edges
    size = min(size, frame_shape[1] - new_x, frame_shape[0] - new_y)

    return (new_x, new_y, size, size)


def detect_roi_color_based(frame):
    """
    Main function: detects all possible sign regions in a frame using color.
    
    Pipeline:
      1. detect_color_regions → get a mask of red/blue/yellow pixels
      2. extract_contours → get outlines of those colored blobs
      3. filter_contours_by_shape → remove non-sign-shaped blobs
      4. make_square_bbox → convert each bbox to square for the model
    
    Returns a list of square bounding boxes where signs might be.
    """
    mask = detect_color_regions(frame)
    contours = extract_contours(mask)
    bboxes = filter_contours_by_shape(contours, frame.shape)

    # Convert every detected bounding box to a square
    square_bboxes = []
    for bbox in bboxes:
        square_bbox = make_square_bbox(*bbox, frame.shape)
        square_bboxes.append(square_bbox)

    return square_bboxes


def get_smart_region_candidates(frame, max_candidates=12):
    """
    Fallback strategy when color-based detection finds nothing.
    
    Some signs don't have strong red/blue/yellow — they may be faded,
    poorly lit, or photographed straight-on. In that case, we try multiple
    cropped regions of the image: the center, smaller crops, and corners.
    
    The model then checks all of them and picks the most confident result.
    This is slower but helps when color detection fails.
    """
    height, width = frame.shape[:2]
    candidates = []
    base_size = min(height, width, 256)

    # Always include the full frame first — if someone uploaded an image
    # of just a sign, the whole image IS the sign
    candidates.append((0, 0, width, height))

    # Try center crops at different zoom levels (100%, 75%, 50%)
    for scale in [1.0, 0.75, 0.5]:
        size = int(base_size * scale)
        if size < 48:
            continue  # Too small to be useful
        x = (width - size) // 2
        y = (height - size) // 2
        if x >= 0 and y >= 0 and x + size <= width and y + size <= height:
            candidates.append((x, y, size, size))

    # Also check the 4 corners — signs are sometimes at the edges of the frame
    grid_size = min(width, height) // 2
    if grid_size >= 64:
        for gy in [0, height - grid_size]:
            for gx in [0, width - grid_size]:
                gx = max(0, min(gx, width - grid_size))
                gy = max(0, min(gy, height - grid_size))
                candidates.append((int(gx), int(gy), grid_size, grid_size))

    # Remove duplicates (boxes that are almost identical)
    seen = set()
    unique = []
    for c in candidates:
        k = (c[0] // 16, c[1] // 16, c[2], c[3])
        if k not in seen:
            seen.add(k)
            unique.append(c)
        if len(unique) >= max_candidates:
            break

    # If something went wrong and we have nothing, default to full frame
    if not unique:
        unique = [(0, 0, width, height)]
    return unique


def sliding_window_detection(frame, window_sizes=[64, 96, 128], step_size=32):
    """
    Alternative fallback: scan the entire image with small sliding windows.
    
    This is computationally expensive so it's not used in the main pipeline,
    but it's here as a utility if needed for more thorough scanning.
    """
    candidates = []
    height, width = frame.shape[:2]

    # Slide a window of each size across the image
    for win_size in window_sizes:
        for y in range(0, height - win_size, step_size):
            for x in range(0, width - win_size, step_size):
                candidates.append((x, y, win_size, win_size))

    return candidates


def extract_roi(frame, bbox):
    """
    Crops out a region of interest from the frame using a bounding box.
    This gives us just the sign portion to pass into the CNN model.
    """
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    return roi


def draw_detection(frame, bbox, label, confidence, draw_bbox=True, mask_background=False):
    """
    Draws the detection result on the frame for the user to see.
    
    - Green box = high confidence (above the threshold)
    - Orange box = low confidence (showing a "best guess")
    - Label and confidence score shown above the box
    """
    x, y, w, h = bbox

    if mask_background:
        # Black out everything except the detected region
        # (not used in main flow but useful for debugging)
        mask = np.zeros_like(frame)
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        frame = mask

    if draw_bbox:
        # Green = confident detection, Orange = uncertain best guess
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)

        # Draw the rectangle around the detected region
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Prepare the label text: "Stop Sign: 0.87"
        label_text = f"{label}: {confidence:.2f}"

        # Calculate how much space the text needs
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )

        # Draw a colored rectangle behind the text so it's readable on any background
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline - 5),
            (x + text_width, y),
            color,
            -1  # -1 means fill the rectangle
        )

        # Draw the white text on top of the colored background
        cv2.putText(
            frame,
            label_text,
            (x, y - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )

    return frame


def non_max_suppression(bboxes, overlap_threshold=0.3):
    """
    Removes duplicate/overlapping bounding boxes, keeping only the best one.
    
    When multiple colored blobs are detected very close together (e.g. the red border
    and the red inner areas of one sign), they'd produce overlapping boxes.
    NMS merges them by keeping only the largest box and removing others that
    overlap with it by more than 30% (the overlap_threshold).
    
    Uses IoU (Intersection over Union) to measure how much two boxes overlap.
    """
    if len(bboxes) == 0:
        return []

    # Convert (x, y, w, h) format to (x1, y1, x2, y2) — easier for overlap math
    boxes = []
    for (x, y, w, h) in bboxes:
        boxes.append([x, y, x+w, y+h])
    boxes = np.array(boxes)

    # Get the four corner coordinates for all boxes at once
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by area — we process larger boxes first
    indices = np.argsort(areas)[::-1]

    keep = []
    while len(indices) > 0:
        # Keep the largest remaining box
        i = indices[0]
        keep.append(i)

        # Calculate how much each remaining box overlaps with the one we just kept
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        union = areas[i] + areas[indices[1:]] - intersection

        # IoU = how much the boxes overlap relative to their combined area
        # 0 = no overlap, 1 = identical boxes
        iou = np.where(union > 0, intersection / union, 0.0)

        # Only keep boxes that don't heavily overlap with our chosen box
        indices = indices[1:][iou < overlap_threshold]

    # Convert back to (x, y, w, h) format
    result = []
    for i in keep:
        x, y, x2, y2 = boxes[i]
        result.append((int(x), int(y), int(x2-x), int(y2-y)))

    return result


if __name__ == "__main__":
    print("ROI detection utilities loaded successfully")
    print("Available functions:")
    print("  - detect_roi_color_based(frame)")
    print("  - sliding_window_detection(frame)")
    print("  - extract_roi(frame, bbox)")
    print("  - draw_detection(frame, bbox, label, confidence)")
    print("  - non_max_suppression(bboxes)")
