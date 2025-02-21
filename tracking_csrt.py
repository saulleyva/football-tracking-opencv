import cv2
import csv
from aux_func import initialize_trackers, color_map

### Global Variables
# input_path = "Spain-Japan_crop.mp4"
input_path = "Brazil-Croatia_crop.mp4"
output_csv = 'example.csv'

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame Number', 'Object ID', 'Color', 'Bounding Box'])

cap = cv2.VideoCapture(input_path)
ret, frame = cap.read()

# Steps 1, 2 and 3 from the algorithm in the paper
CSRT_trackers = initialize_trackers(frame.copy())

# Step 4 from the algorithm
frame_number = 0
with open(output_csv, mode='a', newline='') as file:
    writer = csv.writer(file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        success, boxes = CSRT_trackers.update(frame)

        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, color_map.get(i, (255, 255, 255)), 2, 1)

            # Prepare data for writing
            obj_id = i
            color = color_map.get(i, (255, 255, 255))
            bbox_str = f"{p1[0]}, {p1[1]}, {p2[0] - p1[0]}, {p2[1] - p1[1]}"  # left, top, width, height
            
            # Write data to CSV
            writer.writerow([frame_number, obj_id, color, bbox_str])

        frame_number+=1

        cv2.imshow('Current Frame', frame)
        cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()