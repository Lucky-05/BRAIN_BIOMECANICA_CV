from ultralytics import YOLO
import cv2
import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = YOLO('yolov8n-pose.pt')
model.to(device)

# Open the video file
# 0 depth 
# 1 rgb
cap = cv2.VideoCapture(1)



# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
# out = cv2.VideoWriter('output_video2-+.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

bbox_color = (150, 0, 0)    
bbox_thickness = 6

bbox_labelstr = {
    'font_size':6,       
    'font_thickness':14,  
    'offset_x':0,         
    'offset_y':0,       
}

# BGR
kpt_color_map = {
    # 0:{'name':'Nose', 'color':[0, 0, 255], 'radius':10},               
    # 1:{'name':'Right Eye', 'color':[255, 0, 0], 'radius':10},          
    # 2:{'name':'Left Eye', 'color':[255, 0, 0], 'radius':10},            
    # 3:{'name':'Right Ear', 'color':[0, 255, 0], 'radius':10},        
    # 4:{'name':'Left Ear', 'color':[0, 255, 0], 'radius':10},            
    5:{'name':'Right Shoulder', 'color':[193, 182, 255], 'radius':10},
    6:{'name':'Left Shoulder', 'color':[193, 182, 255], 'radius':10},   
    7:{'name':'Right Elbow', 'color':[16, 144, 247], 'radius':10},     
    8:{'name':'Left Elbow', 'color':[16, 144, 247], 'radius':10},      
    9:{'name':'Right Wrist', 'color':[1, 240, 255], 'radius':10},       
    10:{'name':'Left Wrist', 'color':[1, 240, 255], 'radius':10},       
    11:{'name':'Right Hip', 'color':[140, 47, 240], 'radius':10},       
    12:{'name':'Left Hip', 'color':[140, 47, 240], 'radius':10},       
    13:{'name':'Right Knee', 'color':[223, 155, 60], 'radius':10},      
    14:{'name':'Left Knee', 'color':[223, 155, 60], 'radius':10},       
    15:{'name':'Right Ankle', 'color':[139, 0, 0], 'radius':10},       
    16:{'name':'Left Ankle', 'color':[139, 0, 0], 'radius':10},     
}

skeleton_map = [
    {'srt_kpt_id':15, 'dst_kpt_id':13, 'color':[0, 100, 255], 'thickness':5},       
    {'srt_kpt_id':13, 'dst_kpt_id':11, 'color':[0, 255, 0], 'thickness':5},         
    {'srt_kpt_id':16, 'dst_kpt_id':14, 'color':[255, 0, 0], 'thickness':5},         
    {'srt_kpt_id':14, 'dst_kpt_id':12, 'color':[0, 0, 255], 'thickness':5},         
    {'srt_kpt_id':11, 'dst_kpt_id':12, 'color':[122, 160, 255], 'thickness':5},     
    {'srt_kpt_id':5, 'dst_kpt_id':11, 'color':[139, 0, 139], 'thickness':5},        
    {'srt_kpt_id':6, 'dst_kpt_id':12, 'color':[237, 149, 100], 'thickness':5},      
    {'srt_kpt_id':5, 'dst_kpt_id':6, 'color':[152, 251, 152], 'thickness':5},       
    {'srt_kpt_id':5, 'dst_kpt_id':7, 'color':[148, 0, 69], 'thickness':5},          
    {'srt_kpt_id':6, 'dst_kpt_id':8, 'color':[0, 75, 255], 'thickness':5},          
    {'srt_kpt_id':7, 'dst_kpt_id':9, 'color':[56, 230, 25], 'thickness':5},         
    {'srt_kpt_id':8, 'dst_kpt_id':10, 'color':[0,240, 240], 'thickness':5},         
    # {'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[224,255, 255], 'thickness':5},        
    # {'srt_kpt_id':0, 'dst_kpt_id':1, 'color':[47,255, 173], 'thickness':5},         
    # {'srt_kpt_id':0, 'dst_kpt_id':2, 'color':[203,192,255], 'thickness':5},         
    # {'srt_kpt_id':1, 'dst_kpt_id':3, 'color':[196, 75, 255], 'thickness':5},        
    # {'srt_kpt_id':2, 'dst_kpt_id':4, 'color':[86, 0, 25], 'thickness':5},           
    # {'srt_kpt_id':3, 'dst_kpt_id':5, 'color':[255,255, 0], 'thickness':5},          
    # {'srt_kpt_id':4, 'dst_kpt_id':6, 'color':[255, 18, 200], 'thickness':5}         
]

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform the pose detection on the frame
    results = model(frame, conf=0.7)
    
    num_bbox = len(results[0].boxes.cls)
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy().astype('uint32')
    
    for idx in range(num_bbox):
        bbox_xyxy = bboxes_xyxy[idx] 
        bbox_label = results[0].names[0]
        frame = cv2.rectangle(frame, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)
        frame = cv2.putText(frame, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])
        
        bbox_keypoints = bboxes_keypoints[idx]
        # for skeleton in skeleton_map:
        #     srt_kpt_id = skeleton['srt_kpt_id']
        #     srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
        #     srt_kpt_y = bbox_keypoints[srt_kpt_id][1]
            
        #     dst_kpt_id = skeleton['dst_kpt_id']
        #     dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
        #     dst_kpt_y = bbox_keypoints[dst_kpt_id][1]
            
        #     skeleton_color = skeleton['color']
        #     skeleton_thickness = skeleton['thickness']
            
        #     frame = cv2.line(frame, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color, thickness=skeleton_thickness)
        
        for kpt_id in kpt_color_map:
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]
            frame = cv2.circle(frame, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

    cv2.imshow("Body", frame)
    cv2.waitKey(1)
    # Write the processed frame to the output video
    # out.write(frame)

# Release resources
cap.release()
# out.release()
cv2.destroyAllWindows()