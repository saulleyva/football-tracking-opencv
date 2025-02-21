import cv2
import numpy as np

color_map = {  # Colors for a maximum of 10 different players
    0: (0, 0, 255),  # Red
    1: (0, 255, 0),  # Green
    2: (255, 0, 0),  # Blue
    3: (0, 255, 255),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (255, 255, 0),  # Cyan
    6: (255, 255, 255),  # White
    7: (0, 0, 0),  # Black
    8: (128, 0, 0),  # Maroon
    9: (128, 0, 128),  # Purple
}

def initialize_trackers(frame):
    """
    Lets the user draw multiple bounding boxes on the given frame and initializes trackers for those boxes.
    Press "Enter" to finish selecting boxes.

    :param frame: The first frame of the video where trackers are to be initialized.
    :return: A MultiTracker object with initialized trackers for each selected bounding box.
    """
    trackers = cv2.legacy.MultiTracker_create()
    bboxes = []

    print("Draw bounding boxes on the frame. Press 'Enter' to finish selection.")
    
    while True:
        bbox = cv2.selectROI('Frame', frame, False)
        if bbox[2] != 0 and bbox[3] != 0: 
            bboxes.append(bbox)
            # Draw all selected bounding boxes on the frame
            for box in bboxes:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            cv2.imshow('Frame', frame)  
        else:
            break  
    
    cv2.destroyWindow('Frame') 
    
    # Initialize trackers for all collected bboxes
    for bbox in bboxes:
        tracker = cv2.legacy.TrackerCSRT_create()
        trackers.add(tracker, frame, bbox)
        
    return trackers

def string_to_color(color_string):
    """Convert a string representation of a color to a tuple of integers."""
    color = color_string.strip("()").split(",")
    return (int(color[0]), int(color[1]), int(color[2]))

def euclidean_dist(p1, p2):
    """Return the euclidean distance between two the points (x1, y1) and (x2, y2)."""
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))

def interpolate(x, x0, y0, x1, y1):
    """
    Interpolates a value given x.

    Parameters:
    - x: The position which to interpolate.
    - x0: The y-value of the first point.
    - y0: The real distance of the first point.
    - x1: The y-value of the second point.
    - y1: The real distance of the second point.

    Returns:
    - The interpolated y value.
    """
    # Ensure x1 != x0 to avoid division by zero
    if x1 != x0:
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    else:
        return y0
    
def calculate_activity_count(current_frame, previous_frame, current_bbox, previous_bbox, threshold = 0.99):
    """
    Calculate the activity count based on changes in the ROI between current and previous frames.
    
    Parameters:
    - current_frame: The current frame (grayscale)
    - previous_frame: The previous frame (grayscale)
    - current_bbox: The current bounding box (x, y, width, height)
    - previous_bbox: The previous bounding box (x, y, width, height)
    
    Returns:
    - Activity count (1 for activity detected, 0 otherwise)
    """
    current_roi = current_frame[(current_bbox[1]+int(current_bbox[3]*0.75)):(current_bbox[1]+current_bbox[3]-2), (current_bbox[0]+2):(current_bbox[0]+current_bbox[2]-2)]
    previous_roi = previous_frame[(previous_bbox[1]+int(previous_bbox[3]*0.75)):(previous_bbox[1]+previous_bbox[3]-2), (previous_bbox[0]+2):(previous_bbox[0]+previous_bbox[2]-2)]
    
    current_hist = cv2.calcHist([current_roi], [0], None, [256], [0, 256])
    previous_hist = cv2.calcHist([previous_roi], [0], None, [256], [0, 256])
    
    current_hist = cv2.normalize(current_hist, current_hist).flatten()
    previous_hist = cv2.normalize(previous_hist, previous_hist).flatten()
    
    similarity = cv2.compareHist(current_hist, previous_hist, cv2.HISTCMP_CORREL)
    
    if similarity < threshold:  
        return 1
    else:
        return 0
