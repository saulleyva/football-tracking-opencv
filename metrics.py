import cv2
import numpy as np
import csv
from aux_func import string_to_color, euclidean_dist, interpolate, calculate_activity_count

### Global Variables
cap = cv2.VideoCapture('Spain-Japan_crop.mp4')
# cap = cv2.VideoCapture('Brazil-Croatia_crop.mp4')
csv_file = 'Spain.csv'
output_video_path = "Spain.avi"


# Initialize variables
points = []
frame_for_display = None
colors = []
real_distance = []
selecting = True

# Function to select points on the first frame
def select_points(event, x, y, flags, params):
    global points, frame_for_display, selecting
    if event == cv2.EVENT_LBUTTONDOWN and selecting:
        points.append((x, y))
        cv2.circle(frame_for_display, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(window_name, frame_for_display)
        if len(points) % 2 == 0:  # If two points have been selected
            selecting = False

# Read the first frame
ret, first_frame = cap.read()

frame_for_display = first_frame.copy()
window_name = 'First Frame - Select Points (pair by pair and enter numbers in terminal).'
print("Select 4 pairs of points in the following order:")
print("First, select one horizontal line in the background.")
print("Second, select one horizontal line in the foreground.")
print("Third, select one vertical line in the background.")
print("Finally, select one vertical line in the foreground.")

# Display the first frame to the user
cv2.imshow(window_name, frame_for_display)
cv2.setMouseCallback(window_name, select_points)

while len(real_distance) < 4:
    if not selecting:  # If not currently in selecting mode
        pair_number = input(f"Enter the real distance for the pair of points {len(real_distance) + 1}: ")
        real_distance.append(pair_number)
        selecting = True  # Ready to select the next pair
        if len(real_distance) < 4:  # Check if we need to select more pairs
            print("Select the next pair of points.")
            cv2.imshow(window_name, frame_for_display)
        else:
            print("All required pairs have been selected.")
    cv2.waitKey(100) 

cv2.destroyAllWindows()

# Generate random colors for each pair
num_pairs = len(points) // 2
colors = [np.random.randint(0, 255, size=(3)).tolist() for _ in range(num_pairs)]

# Convert points to a NumPy array of float32, necessary for the calcOpticalFlowPyrLK function
old_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(10, 10),
                 maxLevel=1,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Convert the first frame to grayscale
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
current_frame_number = 0
previous_frame = first_frame

height, width, layers = first_frame.shape
video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

# Read the bounding box details from the CSV file
bbox_details_by_frame = {}
prev_bbox = {}
activity_counts = {}
cross_distances = {}

with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        frame_number = int(row['Frame Number'])
        obj_id = row['Object ID']
        color = string_to_color(row['Color'])
        bbox = tuple(map(int, row['Bounding Box'].split(",")))
        if frame_number not in bbox_details_by_frame:
            bbox_details_by_frame[frame_number] = []
        bbox_details_by_frame[frame_number].append((obj_id, color, bbox))
        activity_counts[obj_id] = 0
        cross_distances[obj_id] = 0
        prev_bbox[obj_id] = bbox

# Initialize variables for storing the last calculated values
last_calculated_ppm_pairs = [0] * num_pairs
last_calculated_speeds = {}
initial_position = {}
final_position = {}
next_initial_position = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    aux_frame = frame.copy()

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Update points using Lucas-Kanade optical flow
    new_points, status, error = cv2.calcOpticalFlowPyrLK(first_frame_gray, frame_gray, old_points, None, **lk_params)

    ppm_midpoint = []

    # Compute ppm for each pair of points
    for i in range(0, len(new_points), 2):
        x1, y1 = new_points[i][0].ravel()
        x2, y2 = new_points[i+1][0].ravel()

        center1, center2 = (int(x1), int(y1)), (int(x2), int(y2))
        color = colors[i // 2]
        cv2.circle(frame, center1, 5, color, -1)
        cv2.circle(frame, center2, 5, color, -1)
        cv2.line(frame, center1, center2, color, 2)

        midpoint = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)

        # Calculate the ppm pair for the current pair of points
        ppm_pair =  euclidean_dist((x1, y1), (x2, y2)) / float(real_distance[i // 2])
        last_calculated_ppm_pairs[i // 2] = ppm_pair  # Update the stored ppm_pair
        ppm_midpoint.append((ppm_pair, midpoint))

        # Retrieve the last calculated ppm_pair and display it
        ppm_pair_to_display = last_calculated_ppm_pairs[i // 2]
        cv2.putText(frame, f'{float(real_distance[i // 2]):.2f}m ({ppm_pair_to_display:.2f}ppm) ', midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Display and calculate the bbox information
    if current_frame_number in bbox_details_by_frame:
        for detail in bbox_details_by_frame[current_frame_number]:
            obj_id, color, bbox = detail
            x, y, width, height = bbox
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

            # Calculate ppm estimation for both axis
            ppm_horizontal = interpolate(y, ppm_midpoint[0][1][1], ppm_midpoint[0][0], ppm_midpoint[1][1][1], ppm_midpoint[1][0])
            ppm_vertical = interpolate(y, ppm_midpoint[2][1][1], ppm_midpoint[2][0], ppm_midpoint[3][1][1], ppm_midpoint[3][0])

            # Cross distance in meters
            if current_frame_number > 0:
                distance_moved_horizontal = abs(bbox[0] - prev_bbox[obj_id][0]) / ppm_horizontal
                distance_moved_vertical = abs(bbox[1] - prev_bbox[obj_id][1]) / ppm_vertical
                cross_distances[obj_id] += np.sqrt(distance_moved_horizontal ** 2 + distance_moved_vertical ** 2)

            if current_frame_number % 30 == 0:
                # Update positions for speed calculation
                final_position[obj_id] = (x, y)
                initial_position[obj_id] = next_initial_position.get(obj_id, (x, y))
                next_initial_position[obj_id]  = (x, y)

                # Calculate the distance moved by the object
                distance_moved_horizontal = abs(final_position[obj_id][0] - initial_position[obj_id][0]) / ppm_horizontal
                distance_moved_vertical = abs(final_position[obj_id][1] - initial_position[obj_id][1]) / ppm_vertical
                distance_moved = np.sqrt(distance_moved_horizontal ** 2 + distance_moved_vertical ** 2) 

                last_calculated_speeds[obj_id] = distance_moved / 1  # 30 frames per second

            # Speed
            speed_to_display = last_calculated_speeds.get(obj_id, 0) 
            # Activity count
            activity_counts[obj_id] += calculate_activity_count(frame, previous_frame, bbox, prev_bbox[obj_id])
        
            prev_bbox[obj_id] = bbox 

            cv2.putText(frame, f'{int(cross_distances[obj_id])}m {activity_counts[obj_id]}Act', (x - 20, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, f'{speed_to_display:.2f}m/s', (x - 20, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(frame, str(current_frame_number), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)
    cv2.waitKey(100)
    video.write(frame)

    # Update the previous frame, previous points and frame number
    first_frame_gray = frame_gray.copy()
    old_points = new_points
    current_frame_number += 1
    previous_frame = aux_frame

cap.release()
video.release()
cv2.destroyAllWindows()
