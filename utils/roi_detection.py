"""
ROI (Region of Interest) detection for traffic signs
Uses color-based segmentation and contour detection
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
    Detect red, blue, and yellow regions in the frame using HSV color space.
    Uses expanded ranges for robustness to lighting and faded signs.
    Args:
        frame: input frame (BGR)
    Returns: combined mask of sign-colored regions
    """
    # Detect red, blue, and yellow regions.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color detection (two ranges because red wraps around in HSV)
    red_mask1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
    red_mask2 = cv2.inRange(hsv, HSV_RED_LOWER2, HSV_RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Blue color detection
    blue_mask = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    
    # Yellow color detection (warning signs)
    yellow_mask = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, blue_mask), yellow_mask)
    
    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask


def extract_contours(mask):
    """
    Extract contours from binary mask
    Args:
        mask: binary mask
    Returns: list of contours
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours_by_shape(contours, frame_shape):
    """
    Filter contours by area and shape characteristics
    Args:
        contours: list of contours
        frame_shape: shape of the frame
    Returns: filtered list of bounding boxes [(x, y, w, h), ...]
    """
    bounding_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (traffic signs are roughly square or circular;
        # allow slightly wider range for rectangular signs like Axle Load Limit)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue
        
        # Filter by extent (ratio of contour area to bounding box area)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        if extent < 0.3:  # Too sparse
            continue
        
        bounding_boxes.append((x, y, w, h))
    
    return bounding_boxes


def make_square_bbox(x, y, w, h, frame_shape):
    """
    Convert bounding box to square by expanding the smaller dimension
    Args:
        x, y, w, h: bounding box coordinates
        frame_shape: shape of the frame (height, width, channels)
    Returns: square bounding box (x, y, size, size)
    """
    # Use the larger dimension
    size = max(w, h)
    
    # Center the square on the original bbox
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate new top-left corner
    new_x = max(0, center_x - size // 2)
    new_y = max(0, center_y - size // 2)
    
    # Ensure bbox doesn't exceed frame boundaries
    if new_x + size > frame_shape[1]:
        new_x = frame_shape[1] - size
    if new_y + size > frame_shape[0]:
        new_y = frame_shape[0] - size
    
    # Ensure coordinates are non-negative
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    
    # Adjust size if necessary
    size = min(size, frame_shape[1] - new_x, frame_shape[0] - new_y)
    
    return (new_x, new_y, size, size)


def detect_roi_color_based(frame):
    """
    Detect ROIs using color-based segmentation
    Args:
        frame: input frame (BGR)
    Returns: list of ROI bounding boxes [(x, y, w, h), ...]
    """
    # Detect color regions
    mask = detect_color_regions(frame)
    
    # Extract contours
    contours = extract_contours(mask)
    
    # Filter contours
    bboxes = filter_contours_by_shape(contours, frame.shape)
    
    # Convert to square bounding boxes
    square_bboxes = []
    for bbox in bboxes:
        square_bbox = make_square_bbox(*bbox, frame.shape)
        square_bboxes.append(square_bbox)
    
    return square_bboxes


def get_smart_region_candidates(frame, max_candidates=12):
    """
    Smart fallback when color-based ROI fails. Returns overlapping regions
    (center crops + grid) so the model can find signs anywhere in the image.
    Good for uploaded images where the sign may not have typical red/blue colors.
    Args:
        frame: input frame
        max_candidates: limit number of regions to try (for performance)
    Returns: list of bounding boxes [(x, y, w, h), ...]
    """
    height, width = frame.shape[:2]
    candidates = []
    base_size = min(height, width, 256)

    # ALWAYS include the full frame first — critical for uploaded images where
    # the sign fills most or all of the frame (e.g. Cattle, Axle Load Limit)
    candidates.append((0, 0, width, height))

    # Center crop at multiple scales
    for scale in [1.0, 0.75, 0.5]:
        size = int(base_size * scale)
        if size < 48:
            continue
        x = (width - size) // 2
        y = (height - size) // 2
        if x >= 0 and y >= 0 and x + size <= width and y + size <= height:
            candidates.append((x, y, size, size))

    # 2x2 grid (covers corners - signs often at edges)
    grid_size = min(width, height) // 2
    if grid_size >= 64:
        for gy in [0, height - grid_size]:
            for gx in [0, width - grid_size]:
                gx = max(0, min(gx, width - grid_size))
                gy = max(0, min(gy, height - grid_size))
                candidates.append((int(gx), int(gy), grid_size, grid_size))

    # Deduplicate and limit
    seen = set()
    unique = []
    for c in candidates:
        k = (c[0] // 16, c[1] // 16, c[2], c[3])
        if k not in seen:
            seen.add(k)
            unique.append(c)
        if len(unique) >= max_candidates:
            break

    # Fallback: full frame
    if not unique:
        unique = [(0, 0, width, height)]
    return unique


def sliding_window_detection(frame, window_sizes=[64, 96, 128], step_size=32):
    """
    Fallback sliding window detection
    Args:
        frame: input frame
        window_sizes: list of window sizes to try
        step_size: step size for sliding
    Returns: list of candidate windows [(x, y, w, h), ...]
    """
    candidates = []
    height, width = frame.shape[:2]
    
    for win_size in window_sizes:
        for y in range(0, height - win_size, step_size):
            for x in range(0, width - win_size, step_size):
                candidates.append((x, y, win_size, win_size))
    
    return candidates


def extract_roi(frame, bbox):
    """
    Extract ROI from frame given bounding box
    Args:
        frame: input frame
        bbox: bounding box (x, y, w, h)
    Returns: cropped ROI
    """
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    return roi


def draw_detection(frame, bbox, label, confidence, draw_bbox=True, mask_background=False):
    """
    Draw detection on frame
    Args:
        frame: input frame
        bbox: bounding box (x, y, w, h)
        label: detected label text
        confidence: confidence score
        draw_bbox: whether to draw bounding box
        mask_background: whether to mask background (show only ROI)
    Returns: annotated frame
    """
    x, y, w, h = bbox
    
    if mask_background:
        # Create a black mask
        mask = np.zeros_like(frame)
        # Copy only the ROI
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        frame = mask
    
    if draw_bbox:
        # Draw bounding box
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label and confidence
        label_text = f"{label}: {confidence:.2f}"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline - 5),
            (x + text_width, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label_text,
            (x, y - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    return frame


def non_max_suppression(bboxes, overlap_threshold=0.3):
    """
    Apply non-maximum suppression to remove overlapping bounding boxes
    Args:
        bboxes: list of bounding boxes [(x, y, w, h), ...]
        overlap_threshold: IoU threshold for suppression
    Returns: filtered list of bounding boxes
    """
    if len(bboxes) == 0:
        return []
    
    # Convert to (x1, y1, x2, y2) format
    boxes = []
    for (x, y, w, h) in bboxes:
        boxes.append([x, y, x+w, y+h])
    boxes = np.array(boxes)
    
    # Calculate areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by area (larger boxes first)
    indices = np.argsort(areas)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Keep the largest box
        i = indices[0]
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[indices[1:]] - intersection
        # Avoid division by zero (e.g. identical/zero-area boxes)
        iou = np.where(union > 0, intersection / union, 0.0)
        
        # Keep boxes with low IoU
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
