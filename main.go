package main

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"sort"
	"math"
	"runtime"
	"strings"
	"sync"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"github.com/sirupsen/logrus"
)

// Constants
const (
	ModelPath     = "yolov8n.onnx"
	ConfThreshold = 0.5
	IOUThreshold  = 0.7
	ImageSize     = 640
)

// Detection represents a single object detection result
type Detection struct {
	ID       string   `json:"id"`
	Result   []Result `json:"result"`
	Score    float32  `json:"score"`
	FromName string   `json:"from_name"`
	ToName   string   `json:"to_name"`
	Type     string   `json:"type"`
	Origin   string   `json:"origin"`
}

// Result represents the bounding box and label information
type Result struct {
	X      float64 `json:"x"`
	Y      float64 `json:"y"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
	Label  string  `json:"label"`
}

// ModelSession holds the ONNX runtime session and tensors
type ModelSession struct {
	sync.RWMutex
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

var (
	model  ModelSession
	once   sync.Once
	logger *logrus.Logger
)

// getSharedLibPath returns the absolute path to the ONNXRuntime shared library.
func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so"
		}
		return "./third_party/onnxruntime.so"
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}

// initLogger initializes Logrus to log to both stdout and a file.
func initLogger() *logrus.Logger {
	log := logrus.New()
	log.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
	})
	// Open a file for logging.
	file, err := os.OpenFile("app.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Warn("Failed to log to file, using default stderr")
	} else {
		// Write to both stdout and file.
		log.SetOutput(io.MultiWriter(os.Stdout, file))
	}
	return log
}

func main() {
	// Initialize logger.
	logger = initLogger()
	logger.Info("Logger initialized")

	// Initialize model on startup.
	if err := initializeModel(); err != nil {
		logger.Fatalf("Failed to initialize model: %v", err)
	}
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, "index.html") // Save the HTML artifact as index.html
	})

	http.HandleFunc("/detect", handleDetection)
	logger.Info("Server starting on :8000")
	if err := http.ListenAndServe(":8000", nil); err != nil {
		logger.Fatalf("Server failed: %v", err)
	}
}

func initializeModel() error {
	var err error
	once.Do(func() {
		err = initYOLOSession()
		if err != nil {
			logger.Errorf("initYOLOSession error: %v", err)
		} else {
			logger.Info("Model session initialized successfully")
		}
	})
	return err
}

func handleDetection(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	tempDir, err := os.MkdirTemp("", "detection")
	if err != nil {
		http.Error(w, "Failed to create temp directory", http.StatusInternalServerError)
		return
	}
	defer os.RemoveAll(tempDir)

	results := processImages(r, tempDir)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(results); err != nil {
		logger.Errorf("Failed to encode JSON response: %v", err)
	}
}

func processImages(r *http.Request, tempDir string) map[string][]Detection {
	var wg sync.WaitGroup
	results := make(map[string][]Detection)
	resultsMutex := sync.RWMutex{}

	for key, fileHeaders := range r.MultipartForm.File {
		if !strings.HasPrefix(key, "camera_") || len(fileHeaders) == 0 {
			continue
		}

		wg.Add(1)
		go func(key string, header *multipart.FileHeader) {
			defer wg.Done()

			file, err := header.Open()
			if err != nil {
				logger.Errorf("Error opening file %s: %v", key, err)
				return
			}
			defer file.Close()

			detections, err := detectObjects(file)
			if err != nil {
				logger.Errorf("Error detecting objects in %s: %v", key, err)
				return
			}

			resultsMutex.Lock()
			results[key] = detections
			resultsMutex.Unlock()
		}(key, fileHeaders[0])
	}

	wg.Wait()
	return results
}

func detectObjects(reader io.Reader) ([]Detection, error) {
	input, imgWidth, imgHeight, err := prepareInput(reader)
	if err != nil {
		logger.Errorf("prepareInput error: %v", err)
		return nil, err
	}

	model.RLock()
	output, err := runInference(input)
	model.RUnlock()

	if err != nil {
		return nil, fmt.Errorf("inference error: %v", err)
	}

	return processOutput(output, imgWidth, imgHeight), nil
}

func prepareInput(reader io.Reader) ([]float32, int64, int64, error) {
	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to decode image: %w", err)
	}
	size := img.Bounds().Size()
	imgWidth, imgHeight := int64(size.X), int64(size.Y)

	resized := resize.Resize(ImageSize, ImageSize, img, resize.Lanczos3)
	input := make([]float32, ImageSize*ImageSize*3)

	idx := 0
	for y := 0; y < ImageSize; y++ {
		for x := 0; x < ImageSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			input[idx] = float32(r>>8) / 255.0
			input[idx+ImageSize*ImageSize] = float32(g>>8) / 255.0
			input[idx+2*ImageSize*ImageSize] = float32(b>>8) / 255.0
			idx++
		}
	}

	return input, imgWidth, imgHeight, nil
}

func processOutput(output []float32, imgWidth, imgHeight int64) []Detection {
    var detections []Detection
    var boxes []Detection

    // First pass: collect all boxes that meet confidence threshold
    for i := 0; i < 8400; i++ {
        classID, prob := 0, float32(0.0)
        for j := 0; j < 80; j++ {
            if curr := output[8400*(j+4)+i]; curr > prob {
                prob = curr
                classID = j
            }
        }

        if prob < ConfThreshold {
            continue
        }

        xc := output[i]
        yc := output[8400+i]
        w := output[2*8400+i]
        h := output[3*8400+i]

        x := (xc - w/2) / ImageSize * float32(imgWidth)
        y := (yc - h/2) / ImageSize * float32(imgHeight)
        width := w / ImageSize * float32(imgWidth)
        height := h / ImageSize * float32(imgHeight)

        detection := Detection{
            ID:       fmt.Sprintf("detection-%d", i),
            FromName: "object",
            ToName:   "image",
            Type:     "rectanglelabels",
            Origin:   "manual",
            Score:    prob,
            Result: []Result{{
                X:      float64(x),
                Y:      float64(y),
                Width:  float64(width),
                Height: float64(height),
                Label:  yoloClasses[classID],
            }},
        }
        boxes = append(boxes, detection)
    }

    // Sort boxes by confidence score in descending order
    sort.Slice(boxes, func(i, j int) bool {
        return boxes[i].Score > boxes[j].Score
    })

    // Apply NMS
    selected := make([]bool, len(boxes))
    for i := 0; i < len(boxes); i++ {
        if selected[i] {
            continue
        }

        detections = append(detections, boxes[i])
        selected[i] = true

        for j := i + 1; j < len(boxes); j++ {
            if selected[j] {
                continue
            }

            // Calculate IoU between boxes[i] and boxes[j]
            iou := calculateIoU(boxes[i].Result[0], boxes[j].Result[0])
            if iou > IOUThreshold {
                selected[j] = true
            }
        }
    }

    return detections
}
func calculateIoU(box1, box2 Result) float32 {
    // Calculate coordinates of intersection rectangle
    x1 := math.Max(box1.X, box2.X)
    y1 := math.Max(box1.Y, box2.Y)
    x2 := math.Min(box1.X+box1.Width, box2.X+box2.Width)
    y2 := math.Min(box1.Y+box1.Height, box2.Y+box2.Height)

    // Calculate area of intersection rectangle
    intersectionArea := math.Max(0, x2-x1) * math.Max(0, y2-y1)

    // Calculate area of both boxes
    box1Area := box1.Width * box1.Height
    box2Area := box2.Width * box2.Height

    // Calculate IoU
    iou := float32(intersectionArea / (box1Area + box2Area - intersectionArea))
    return iou
}

func runInference(input []float32) ([]float32, error) {
	copy(model.Input.GetData(), input)
	if err := model.Session.Run(); err != nil {
		return nil, err
	}
	return model.Output.GetData(), nil
}

func initYOLOSession() error {
	ort.SetSharedLibraryPath(getSharedLibPath())
	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}

	inputShape := ort.NewShape(1, 3, ImageSize, ImageSize)
	inputTensor, err := ort.NewTensor(inputShape, make([]float32, ImageSize*ImageSize*3))
	if err != nil {
		return err
	}

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return err
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return err
	}
	defer options.Destroy()

	session, err := ort.NewAdvancedSession(
		ModelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		options,
	)
	if err != nil {
		return err
	}

	model = ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}

	return nil
}

// YOLOv8 class labels
var yoloClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
