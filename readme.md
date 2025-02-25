# YOLO Object Detection Service

This is a Go-based HTTP service that performs object detection using YOLOv8 with ONNX Runtime. The service accepts images via HTTP POST requests and returns detection results in JSON format.

## Major Improvement

This process has reduced the time of inference 
for a batch of 10 images
on python it was 1070 ms and from this method it was 214 ms (Marked on CPU Apple Macbook Pro M3 Pro)


## Prerequisites

- Go 1.24 or higher
- ONNX Runtime shared libraries
- YOLOv8 ONNX model
- Git

## Model Export

Before running the service, you need to export your YOLOv8 model to ONNX format. Install `ultralytics` and run:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx simplify=True half=True
```

This will create `yolov8n.onnx` in your current directory.

## Project Setup

1. Clone the repository:
```bash
git clone https://github.com/harshnandwana/yologoinference.git
cd yologoinference
```

2. Install Go dependencies:
```bash
go mod init object-detection-service
go mod tidy
```

Required packages will be installed:
- github.com/nfnt/resize
- github.com/yalue/onnxruntime_go
- github.com/sirupsen/logrus

3. Set up ONNX Runtime:

Create a `third_party` directory in your project root and download the appropriate ONNX Runtime shared library for your system:

```bash
mkdir third_party
cd third_party
```

Download the appropriate library based on your system:
- Windows (AMD64): `onnxruntime.dll`
- Linux (AMD64): `onnxruntime.so`
- Linux (ARM64): `onnxruntime_arm64.so`
- macOS (ARM64): `onnxruntime_arm64.dylib`

4. Place your exported ONNX model:
```bash
cp yolov8n.onnx <project-directory>/
```

## Project Structure

```
.
├── main.go
├── index.html
├── yolov8n.onnx
├── third_party/
│   ├── onnxruntime.dll
│   ├── onnxruntime.so
│   ├── onnxruntime_arm64.so
│   └── onnxruntime_arm64.dylib
├── go.mod
├── go.sum
└── README.md
```

## Building and Running

1. Build the service:
```bash
go build -o object-detection-service
```

2. Run the service:
```bash
./object-detection-service
```

The service will start on port 8000.

## API Usage

Send POST requests to `/detect` endpoint with multipart form data:

```bash
curl -X POST -F "camera_1=@/path/to/image1.jpg" \
            -F "camera_2=@/path/to/image2.jpg" \
            http://localhost:8000/detect
```

### Request Format
- Method: POST
- Endpoint: `/detect`
- Content-Type: `multipart/form-data`
- File fields should be prefixed with `camera_` (e.g., `camera_1`, `camera_2`)

### Response Format
```json
{
  "camera_1": [
    {
      "id": "detection-1",
      "result": [{
        "x": 100,
        "y": 200,
        "width": 50,
        "height": 80,
        "label": "person"
      }],
      "score": 0.95,
      "from_name": "object",
      "to_name": "image",
      "type": "rectanglelabels",
      "origin": "manual"
    }
  ]
}
```

## Configuration

Key constants in `main.go`:
- `ConfThreshold`: Confidence threshold for detections (default: 0.5)
- `IOUThreshold`: IoU threshold for NMS (default: 0.7)
- `ImageSize`: Input image size for the model (default: 640)

## Logging

The service logs to both stdout and `app.log` file. Check these logs for debugging and monitoring.

## Error Handling

The service includes comprehensive error handling and logging. Check the logs for any issues during operation.

## Performance Considerations

- The service processes images concurrently
- Uses read-write mutex for model access
- Implements Non-Maximum Suppression (NMS) for better detection results

## License

[Your License Here]
