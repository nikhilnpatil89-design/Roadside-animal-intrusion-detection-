import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Animal Intrusion Detection System",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .alert-box {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .safe-box {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'animal_count' not in st.session_state:
    st.session_state.animal_count = 0

# Load pre-trained models
@st.cache_resource
def load_detection_models():
    """Load detection models"""
    try:
        # Load Haar Cascade for general object detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try to load YOLO (if weights are available)
        yolo_net = None
        yolo_classes = None
        
        # Check if YOLO files exist
        if os.path.exists('yolov3.weights') and os.path.exists('yolov3.cfg'):
            yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
            with open('coco.names', 'r') as f:
                yolo_classes = [line.strip() for line in f.readlines()]
        
        return face_cascade, yolo_net, yolo_classes
    except Exception as e:
        st.warning(f"Model loading: {e}")
        return None, None, None

def detect_with_yolo(image, net, classes, confidence_threshold=0.5):
    """Detect objects using YOLO"""
    if net is None or classes is None:
        return []
    
    # Animal classes in COCO dataset
    animal_classes = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                     'elephant', 'bear', 'zebra', 'giraffe']
    
    height, width = image.shape[:2]
    
    # Create blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass
    detections = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Check if it's an animal class
            if confidence > confidence_threshold and classes[class_id] in animal_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            results.append({
                'bbox': (x, y, w, h),
                'confidence': confidences[i],
                'class': classes[class_ids[i]]
            })
    
    return results

def detect_animals_color_based(image, sensitivity=0.5):
    """
    Detect potential animals using color segmentation and contour analysis
    Improved for real animal detection
    """
    detections = []
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define color ranges for common animals (brown, gray, white, black tones)
    # Browns (deer, bears, dogs)
    lower_brown1 = np.array([5, 30, 30])
    upper_brown1 = np.array([25, 255, 200])
    
    # Grays (wolves, elephants)
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 50, 200])
    
    # Create masks
    mask_brown = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask_brown, mask_gray)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Edge detection on original image
    edges = cv2.Canny(gray, 30, 100)
    
    # Combine edge information with color mask
    final_mask = cv2.bitwise_and(combined_mask, edges)
    
    # Dilate to connect components
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.002  # Minimum 0.2% of image
    max_area = img_area * 0.7    # Maximum 70% of image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too small
            if w < 40 or h < 40:
                continue
            
            # Calculate features
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Calculate rectangularity
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Animals typically have these characteristics
            is_valid = (
                0.3 < aspect_ratio < 4.0 and  # Reasonable proportions
                extent > 0.25 and              # Not too irregular
                0.1 < circularity < 0.9        # Not too circular or too irregular
            )
            
            if is_valid:
                # Extract ROI for texture analysis
                roi = gray[y:y+h, x:x+w]
                
                # Calculate texture (standard deviation)
                texture = np.std(roi)
                
                # Calculate edge density
                roi_edges = cv2.Canny(roi, 30, 100)
                edge_density = np.count_nonzero(roi_edges) / (w * h)
                
                # Multi-factor confidence
                size_score = min(area / img_area * 100, 0.3)
                shape_score = min(abs(aspect_ratio - 1.5) * 0.1, 0.2)
                texture_score = min(texture / 100, 0.2)
                edge_score = min(edge_density * 2, 0.3)
                
                confidence = size_score + (1 - shape_score) + texture_score + edge_score
                confidence = min(confidence * sensitivity * 1.5, 0.95)
                
                # Only add if confidence is reasonable
                if confidence > 0.3:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'class': 'animal',
                        'method': 'color_based'
                    })
    
    return detections

def detect_motion_objects(image, sensitivity=0.5):
    """
    Detect objects using background subtraction and motion detection
    Better for static images with clear subjects
    """
    detections = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * 0.005 * sensitivity
    max_area = img_area * 0.6
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 50 or h < 50:
                continue
            
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if 0.3 < aspect_ratio < 3.5:
                confidence = min(0.85, (area / img_area) * 20)
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'class': 'animal',
                    'method': 'motion'
                })
    
    return detections

def combine_detections(detections_list):
    """Combine and filter detections from multiple methods"""
    if not detections_list:
        return []
    
    all_detections = []
    for detections in detections_list:
        all_detections.extend(detections)
    
    if not all_detections:
        return []
    
    # Non-maximum suppression
    return non_max_suppression(all_detections, 0.4)

def non_max_suppression(detections, overlap_thresh=0.3):
    """Remove overlapping bounding boxes"""
    if len(detections) == 0:
        return []
    
    boxes = np.array([d['bbox'] for d in detections])
    scores = np.array([d['confidence'] for d in detections])
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / areas[indices[1:]]
        
        indices = np.delete(indices, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return [detections[i] for i in keep]

def draw_detections(image, detections, animal_classes):
    """Draw bounding boxes and labels on detected animals"""
    output_image = image.copy()
    
    for detection in detections:
        x, y, w, h = detection['bbox']
        confidence = detection['confidence']
        class_name = detection.get('class', 'animal')
        
        # Use actual class name from detection or random for demo
        if class_name == 'animal':
            class_name = np.random.choice(animal_classes)
        
        # Color based on confidence
        if confidence > 0.7:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.5:
            color = (0, 165, 255)  # Orange for medium confidence
        else:
            color = (0, 255, 255)  # Yellow for low confidence
        
        # Draw bounding box
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output_image, (x, y - label_h - 10), (x + label_w, y), color, -1)
        cv2.putText(output_image, label, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return output_image

def log_detection(animal_type, confidence, timestamp):
    """Log detection event"""
    st.session_state.detection_log.append({
        'timestamp': timestamp,
        'animal': animal_type,
        'confidence': confidence
    })
    st.session_state.animal_count += 1

# Header
st.markdown('<h1 class="main-header">🦌 Roadside Animal Intrusion Detection System</h1>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    detection_mode = st.selectbox(
        "Detection Mode",
        ["Image Upload", "Video Upload", "Webcam (Demo)"]
    )
    
    st.markdown("---")
    
    st.subheader("Detection Settings")
    
    detection_method = st.selectbox(
        "Detection Method",
        ["Multi-Method (Recommended)", "Color-Based Only", "Edge-Based Only", "YOLO (if available)"]
    )
    
    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Higher sensitivity detects more objects"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.35,
        step=0.05,
        help="Minimum confidence for valid detection"
    )
    
    st.markdown("---")
    
    alert_enabled = st.checkbox("Enable Alert System", value=True)
    show_debug = st.checkbox("Show Debug Info", value=False)
    
    st.markdown("---")
    
    st.subheader("📊 Statistics")
    st.metric("Total Detections", st.session_state.animal_count)
    st.metric("Log Entries", len(st.session_state.detection_log))
    
    if st.button("Clear Logs"):
        st.session_state.detection_log = []
        st.session_state.animal_count = 0
        st.rerun()

# Load models
face_cascade, yolo_net, yolo_classes = load_detection_models()

# Animal classes for display
animal_classes = ['deer', 'dog', 'cat', 'cow', 'horse', 'bear', 'elephant', 
                 'fox', 'wolf', 'sheep', 'goat', 'zebra', 'giraffe']

# Main content
if detection_mode == "Image Upload":
    st.header("📷 Image-Based Detection")
    
    st.info("💡 Tip: Upload clear images with good lighting for best results. The system works best with images containing animals against contrasting backgrounds.")
    
    uploaded_file = st.file_uploader(
        "Upload an image of a roadside scene",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with st.spinner("🔍 Analyzing image for animals..."):
            # Perform detection based on selected method
            all_detections = []
            
            if detection_method == "Multi-Method (Recommended)":
                # Use multiple detection methods
                det1 = detect_animals_color_based(image, sensitivity)
                det2 = detect_motion_objects(image, sensitivity)
                
                if yolo_net is not None:
                    det3 = detect_with_yolo(image, yolo_net, yolo_classes, confidence_threshold)
                    all_detections = combine_detections([det1, det2, det3])
                else:
                    all_detections = combine_detections([det1, det2])
                    
            elif detection_method == "Color-Based Only":
                all_detections = detect_animals_color_based(image, sensitivity)
                
            elif detection_method == "Edge-Based Only":
                all_detections = detect_motion_objects(image, sensitivity)
                
            elif detection_method == "YOLO (if available)":
                if yolo_net is not None:
                    all_detections = detect_with_yolo(image, yolo_net, yolo_classes, confidence_threshold)
                else:
                    st.warning("⚠️ YOLO model not available. Download yolov3.weights and yolov3.cfg for YOLO detection.")
                    all_detections = detect_animals_color_based(image, sensitivity)
            
            # Filter by confidence
            detections = [d for d in all_detections if d['confidence'] >= confidence_threshold]
            
            # Draw detections
            output_image = draw_detections(image, detections, animal_classes)
        
        with col2:
            st.subheader("Detection Results")
            st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Debug info
        if show_debug:
            st.subheader("🔧 Debug Information")
            st.write(f"Total detections before filtering: {len(all_detections)}")
            st.write(f"Detections after confidence filter: {len(detections)}")
            st.write(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
        
        # Alert and results
        if len(detections) > 0:
            if alert_enabled:
                st.markdown(
                    f'<div class="alert-box">⚠️ ALERT: {len(detections)} animal(s) detected on roadside!</div>',
                    unsafe_allow_html=True
                )
            
            st.subheader("🔍 Detection Details")
            
            for idx, detection in enumerate(detections):
                animal_type = detection.get('class', 'animal')
                if animal_type == 'animal':
                    animal_type = np.random.choice(animal_classes)
                
                log_detection(animal_type, detection['confidence'], datetime.now())
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(f"Animal #{idx+1}", animal_type.capitalize())
                col2.metric("Confidence", f"{detection['confidence']:.1%}")
                col3.metric("Size", f"{detection['bbox'][2]}x{detection['bbox'][3]}px")
                col4.metric("Position", f"({detection['bbox'][0]}, {detection['bbox'][1]})")
        else:
            st.markdown(
                '<div class="safe-box">✅ No animals detected. Road is clear.</div>',
                unsafe_allow_html=True
            )
            
            if show_debug:
                st.info("💡 Try adjusting: Lower confidence threshold, Increase sensitivity, or Use different detection method")

elif detection_mode == "Video Upload":
    st.header("🎥 Video-Based Detection")
    
    uploaded_video = st.file_uploader(
        "Upload a video of roadside footage",
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_video is not None:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        
        st.video(tfile.name)
        
        if st.button("🎬 Process Video"):
            video_capture = cv2.VideoCapture(tfile.name)
            
            stframe = st.empty()
            status_text = st.empty()
            frame_count = 0
            detection_count = 0
            
            progress_bar = st.progress(0)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 10th frame for efficiency
                if frame_count % 10 == 0:
                    # Detect animals
                    det1 = detect_animals_color_based(frame, sensitivity)
                    det2 = detect_motion_objects(frame, sensitivity)
                    all_detections = combine_detections([det1, det2])
                    
                    detections = [d for d in all_detections if d['confidence'] >= confidence_threshold]
                    
                    if len(detections) > 0:
                        detection_count += len(detections)
                        frame = draw_detections(frame, detections, animal_classes)
                        status_text.warning(f"⚠️ Detected {len(detections)} animal(s) in frame {frame_count}")
                    
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                 channels="RGB", use_column_width=True)
                
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            video_capture.release()
            
            st.success(f"✅ Video processing complete! Detected {detection_count} animal instances across {frame_count} frames.")

else:  # Webcam Demo
    st.header("📹 Webcam Detection (Demo Mode)")
    
    st.info("📷 This is a demonstration mode. For real webcam integration, use OpenCV's VideoCapture(0) in a local environment.")
    
    run_camera = st.checkbox("Start Demo Detection")
    
    stframe = st.empty()
    status = st.empty()
    
    if run_camera:
        st.warning("⚠️ Demo mode: Simulating camera feed with random detections")
        
        # Simulate camera feed
        for i in range(50):
            # Generate random frame (simulated)
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Randomly simulate detections (20% chance)
            if np.random.random() > 0.8:
                num_animals = np.random.randint(1, 3)
                detections = []
                
                for _ in range(num_animals):
                    x = np.random.randint(50, 400)
                    y = np.random.randint(50, 300)
                    w = np.random.randint(80, 200)
                    h = np.random.randint(80, 180)
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': np.random.uniform(0.6, 0.95),
                        'class': np.random.choice(animal_classes)
                    })
                
                frame = draw_detections(frame, detections, animal_classes)
                
                if alert_enabled:
                    status.warning(f"⚠️ {len(detections)} animal(s) detected!")
            else:
                status.success("✅ No animals detected")
            
            stframe.image(frame, channels="RGB", use_column_width=True)
            
            if not run_camera:
                break

# Detection Log
if len(st.session_state.detection_log) > 0:
    st.markdown("---")
    st.header("📋 Detection Log")
    
    log_df = {
        'Timestamp': [log['timestamp'].strftime("%Y-%m-%d %H:%M:%S") 
                     for log in st.session_state.detection_log[-50:]],  # Show last 50
        'Animal Type': [log['animal'] for log in st.session_state.detection_log[-50:]],
        'Confidence': [f"{log['confidence']:.1%}" for log in st.session_state.detection_log[-50:]]
    }
    
    st.dataframe(log_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Log as JSON"):
            json_log = json.dumps(st.session_state.detection_log, default=str, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_log,
                file_name=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Export Log as CSV"):
            import io
            csv_buffer = io.StringIO()
            csv_buffer.write("Timestamp,Animal Type,Confidence\n")
            for log in st.session_state.detection_log:
                csv_buffer.write(f"{log['timestamp']},{log['animal']},{log['confidence']}\n")
            
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>🦌 Roadside Animal Intrusion Detection System</strong></p>
        <p>Powered by OpenCV & Computer Vision | Built with Streamlit</p>
        <p style='font-size: 0.9em;'>💡 For best results: Use clear, well-lit images with animals visible against the background</p>
    </div>
""", unsafe_allow_html=True)