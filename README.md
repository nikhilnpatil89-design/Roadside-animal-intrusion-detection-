# 🦌 Roadside Animal Intrusion Detection System

A real-time computer vision application for detecting animals near roadways using OpenCV and Deep Learning. Built with Streamlit for an interactive web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Detection Methods](#detection-methods)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance Tips](#performance-tips)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Functionality
- 🖼️ **Image-based Detection** - Upload and analyze static images
- 🎥 **Video Processing** - Detect animals in video footage
- 📹 **Live Camera Feed** - Real-time detection (demo mode included)
- 🎯 **Multi-Method Detection** - Combines multiple CV algorithms for accuracy

### Detection Capabilities
- **Color-Based Detection** - Identifies animals by typical color patterns
- **Edge-Based Detection** - Uses contour analysis and shape recognition
- **YOLO Integration** - Deep learning model support (optional)
- **Multi-Method Fusion** - Combines all methods for best results

### User Interface
- 🎨 Modern, responsive web interface
- ⚙️ Adjustable sensitivity and confidence thresholds
- 📊 Real-time statistics and metrics
- 🚨 Alert system for detected animals
- 📋 Detection logging with export capabilities
- 🐛 Debug mode for troubleshooting

### Data Management
- 📥 Export detection logs as JSON or CSV
- 📈 Visual detection history
- 🔄 Session-based statistics tracking

## 🎬 Demo

### Image Detection
Upload an image containing animals, and the system will:
1. Analyze the image using multiple detection methods
2. Draw bounding boxes around detected animals
3. Display confidence scores and animal classifications
4. Log all detections with timestamps

### Video Processing
Process video files frame-by-frame to:
- Track animals throughout the video
- Count total detections
- Display real-time alerts

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster processing

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/animal-intrusion-detection.git
cd animal-intrusion-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Download YOLO Weights
For enhanced deep learning detection:

```bash
# Download YOLOv3 weights (236 MB)
wget https://pjreddie.com/media/files/yolov3.weights

# Download YOLOv3 configuration
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

# Download COCO class names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

Place these files in the project root directory.

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Basic Workflow

1. **Select Detection Mode** (Sidebar)
   - Image Upload
   - Video Upload
   - Webcam (Demo)

2. **Choose Detection Method** (Sidebar)
   - Multi-Method (Recommended)
   - Color-Based Only
   - Edge-Based Only
   - YOLO (if weights available)

3. **Adjust Parameters**
   - Detection Sensitivity: 0.1 - 1.0 (default: 0.6)
   - Confidence Threshold: 0.1 - 0.9 (default: 0.35)

4. **Upload Media** or **Start Camera**
   - Upload image/video files
   - View real-time detection results

5. **Review Results**
   - View bounding boxes on detected animals
   - Check confidence scores and classifications
   - Export detection logs

### Example Commands

```bash
# Run with default settings
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8080

# Run with specific configuration
streamlit run app.py --server.headless true
```

## 🔍 Detection Methods

### 1. Color-Based Detection
Identifies animals by analyzing color patterns typical of wildlife:
- Brown tones (deer, bears, dogs)
- Gray tones (wolves, elephants)
- Combined with texture analysis

**Best for:** Animals with distinct coloring against background

### 2. Edge-Based Detection
Uses contour analysis and shape recognition:
- Adaptive thresholding
- Morphological operations
- Shape characteristic analysis (aspect ratio, circularity)

**Best for:** Clear subjects with defined edges

### 3. YOLO Detection (Optional)
Deep learning-based object detection:
- Pre-trained on COCO dataset
- Recognizes 80+ object classes
- High accuracy for common animals

**Best for:** Production environments with GPU

### 4. Multi-Method (Recommended)
Combines all available methods:
- Runs multiple algorithms in parallel
- Uses non-maximum suppression to filter duplicates
- Highest overall accuracy

**Best for:** General use cases

## ⚙️ Configuration

### Adjustable Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Detection Sensitivity | 0.1 - 1.0 | 0.6 | Higher = detects smaller objects |
| Confidence Threshold | 0.1 - 0.9 | 0.35 | Minimum confidence for valid detection |
| Alert System | On/Off | On | Enable/disable visual alerts |
| Debug Mode | On/Off | Off | Show detailed detection information |

### Performance Settings

For better performance on large videos:
- Process every Nth frame (default: every 10th frame)
- Lower image resolution before processing
- Use GPU acceleration with YOLO

## 📁 Project Structure

```
animal-intrusion-detection/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── models/                # (Optional) Model weights directory
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── coco.names
│
├── samples/               # Sample images/videos for testing
│   ├── deer.jpg
│   ├── dog.jpg
│   └── road_video.mp4
│
├── logs/                  # Detection logs (auto-generated)
│   └── detection_log_*.json
│
└── docs/                  # Additional documentation
    ├── installation.md
    └── troubleshooting.md
```

## 💡 Performance Tips

### For Best Detection Results:

1. **Image Quality**
   - Use high-resolution images (min 640x480)
   - Ensure good lighting conditions
   - Avoid motion blur in images

2. **Background**
   - Animals should contrast with background
   - Avoid cluttered scenes
   - Open roadside environments work best

3. **Parameter Tuning**
   - Start with sensitivity: 0.6-0.8
   - Lower confidence threshold for more detections: 0.3-0.4
   - Use Multi-Method for balanced results

4. **Hardware Optimization**
   - Use GPU for YOLO detection
   - Process videos at lower frame rates
   - Batch process multiple images

### Troubleshooting

**No detections appearing:**
- Lower confidence threshold to 0.3
- Increase sensitivity to 0.8-0.9
- Try different detection method
- Enable debug mode to see what's happening

**Too many false positives:**
- Increase confidence threshold to 0.5+
- Lower sensitivity to 0.4-0.5
- Use YOLO method if available

**Slow processing:**
- Reduce image/video resolution
- Process fewer video frames
- Close other applications

## 🚀 Future Enhancements

### Planned Features
- [ ] Species classification (specific animal types)
- [ ] Night vision / infrared image support
- [ ] Real-time video streaming from IP cameras
- [ ] Mobile app integration
- [ ] GPS location tagging
- [ ] Email/SMS alert notifications
- [ ] Cloud deployment (AWS/Azure)
- [ ] Multi-camera support
- [ ] Historical data analytics dashboard
- [ ] Integration with wildlife databases

### Advanced Features
- [ ] YOLOv8 integration for better accuracy
- [ ] Custom model training for specific regions
- [ ] Thermal camera support
- [ ] Drone footage analysis
- [ ] 3D depth estimation
- [ ] Animal behavior analysis

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/animal-intrusion-detection.git

# Create development branch
git checkout -b dev

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Streamlit team for the amazing framework
- YOLO authors for the detection model
- Contributors and testers

## 📞 Contact

- Project Link: [https://github.com/yourusername/animal-intrusion-detection](https://github.com/yourusername/animal-intrusion-detection)
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 📊 System Requirements

### Minimum Requirements
- CPU: Dual-core processor
- RAM: 4GB
- Storage: 1GB free space
- OS: Windows 10/Linux/MacOS

### Recommended Requirements
- CPU: Quad-core processor or better
- RAM: 8GB or more
- GPU: NVIDIA GPU with CUDA support
- Storage: 5GB free space (with YOLO weights)
- OS: Windows 10/11, Ubuntu 20.04+, MacOS 11+

## 🔐 Security & Privacy

- All processing is done locally
- No data is sent to external servers
- Detection logs are stored locally
- User has full control over data export/deletion

## 📚 Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [YOLO Documentation](https://pjreddie.com/darknet/yolo/)
- [Computer Vision Tutorials](https://www.pyimagesearch.com/)

---

<div align="center">
Made with ❤️ for wildlife conservation and road safety

**If you find this project helpful, please consider giving it a ⭐!**
</div>
