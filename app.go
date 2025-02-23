// package main

// import (
// 	"encoding/json"
// 	"fmt"
// 	"image"
// 	_ "image/jpeg"
// 	_ "image/png"
// 	"io"
// 	"math"
// 	"mime/multipart"
// 	"net/http"
// 	"os"
// 	"runtime"
// 	"sort"
// 	"strings"
// 	"sync"

// 	"github.com/nfnt/resize"
// 	ort "github.com/yalue/onnxruntime_go"
// 	"github.com/sirupsen/logrus"
// )

// const (
// 	ModelPath     = "/Users/harshnandwana/Desktop/ppe_final/yologoinference/ppe.onnx"
// 	ConfThreshold = 0.5
// 	IOUThreshold  = 0.7
// 	ImageSize     = 640
// 	// MaxPoolSize acts as an upper bound in case 60% of cores is high.
// 	MaxPoolSize = 10
// )

// type Detection struct {
// 	ID       string   `json:"id"`
// 	Result   []Result `json:"result"`
// 	Score    float32  `json:"score"`
// 	FromName string   `json:"from_name"`
// 	ToName   string   `json:"to_name"`
// 	Type     string   `json:"type"`
// 	Origin   string   `json:"origin"`
// }

// type Result struct {
// 	X      float64 `json:"x"`
// 	Y      float64 `json:"y"`
// 	Width  float64 `json:"width"`
// 	Height float64 `json:"height"`
// 	Label  string  `json:"label"`
// }

// type ModelSession struct {
// 	Session *ort.AdvancedSession
// 	Input   *ort.Tensor[float32]
// 	Output  *ort.Tensor[float32]
// }

// type ModelPool struct {
// 	sessions chan *ModelSession
// }

// var (
// 	modelPool *ModelPool
// 	once      sync.Once
// 	logger    *logrus.Logger
// )

// // NewModelPool pre-instantiates sessions based on the given pool size.
// func NewModelPool(size int) (*ModelPool, error) {
// 	pool := &ModelPool{
// 		sessions: make(chan *ModelSession, size),
// 	}
// 	for i := 0; i < size; i++ {
// 		session, err := createModelSession()
// 		if err != nil {
// 			return nil, fmt.Errorf("failed to create model session %d: %v", i, err)
// 		}
// 		pool.sessions <- session
// 	}
// 	return pool, nil
// }

// func (p *ModelPool) GetSession() *ModelSession {
// 	return <-p.sessions
// }

// func (p *ModelPool) ReturnSession(session *ModelSession) {
// 	p.sessions <- session
// }

// func createModelSession() (*ModelSession, error) {
// 	inputShape := ort.NewShape(1, 3, ImageSize, ImageSize)
// 	inputTensor, err := ort.NewTensor(inputShape, make([]float32, ImageSize*ImageSize*3))
// 	if err != nil {
// 		return nil, err
// 	}

// 	outputShape := ort.NewShape(1, 14, 8400)
// 	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
// 	if err != nil {
// 		return nil, err
// 	}

// 	options, err := ort.NewSessionOptions()
// 	if err != nil {
// 		return nil, err
// 	}
// 	defer options.Destroy()

// 	// Restrict threads per session to optimize CPU usage.
// 	options.SetIntraOpNumThreads(1)
// 	options.SetInterOpNumThreads(1)

// 	session, err := ort.NewAdvancedSession(
// 		ModelPath,
// 		[]string{"images"},
// 		[]string{"output0"},
// 		[]ort.ArbitraryTensor{inputTensor},
// 		[]ort.ArbitraryTensor{outputTensor},
// 		options,
// 	)
// 	if err != nil {
// 		return nil, err
// 	}

// 	return &ModelSession{
// 		Session: session,
// 		Input:   inputTensor,
// 		Output:  outputTensor,
// 	}, nil
// }

// func getSharedLibPath() string {
// 	switch runtime.GOOS {
// 	case "windows":
// 		if runtime.GOARCH == "amd64" {
// 			return "./third_party/onnxruntime.dll"
// 		}
// 	case "darwin":
// 		if runtime.GOARCH == "arm64" {
// 			return "./third_party/onnxruntime_arm64.dylib"
// 		}
// 	case "linux":
// 		if runtime.GOARCH == "arm64" {
// 			return "./third_party/onnxruntime_arm64.so"
// 		}
// 		return "./third_party/onnxruntime.so"
// 	}
// 	panic("Unable to find a version of the onnxruntime library supporting this system.")
// }

// func initLogger() *logrus.Logger {
// 	log := logrus.New()
// 	log.SetFormatter(&logrus.TextFormatter{
// 		FullTimestamp: true,
// 	})
// 	file, err := os.OpenFile("app.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
// 	if err != nil {
// 		log.Warn("Failed to log to file, using default stderr")
// 	} else {
// 		log.SetOutput(io.MultiWriter(os.Stdout, file))
// 	}
// 	return log
// }

// func main() {
// 	logger = initLogger()
// 	logger.Info("Logger initialized")

// 	if err := initializeModelPool(); err != nil {
// 		logger.Fatalf("Failed to initialize model pool: %v", err)
// 	}

// 	http.HandleFunc("/detect", handleDetection)
// 	logger.Info("Server starting on :8000")
// 	if err := http.ListenAndServe(":8000", nil); err != nil {
// 		logger.Fatalf("Server failed: %v", err)
// 	}
// }

// func initializeModelPool() error {
// 	var err error
// 	once.Do(func() {
// 		ort.SetSharedLibraryPath(getSharedLibPath())
// 		if err = ort.InitializeEnvironment(); err != nil {
// 			return
// 		}

// 		// Calculate pool size based on available cores (or 60% of them)
// 		cores := runtime.NumCPU()
// 		poolSize := int(math.Round(float64(cores) * 0.8))
// 		if poolSize < 1 {
// 			poolSize = 1
// 		}
// 		if poolSize > MaxPoolSize {
// 			poolSize = MaxPoolSize
// 		}

// 		modelPool, err = NewModelPool(poolSize)
// 		if err != nil {
// 			logger.Errorf("Failed to create model pool: %v", err)
// 			return
// 		}
// 		logger.Infof("Model pool initialized with %d sessions", poolSize)

// 		// Warmup ensures the models are fully loaded in memory
// 		warmUpModelPool(modelPool)
// 	})
// 	return err
// }

// // warmUpSession runs a dummy inference to initialize a session.
// func warmUpSession(session *ModelSession) error {
// 	dummyInput := make([]float32, ImageSize*ImageSize*3) // all zeros
// 	copy(session.Input.GetData(), dummyInput)
// 	return session.Session.Run()
// }

// // warmUpModelPool warms up all sessions in the pool concurrently.
// func warmUpModelPool(pool *ModelPool) {
// 	logger.Info("Warming up model sessions")
// 	sessionCount := len(pool.sessions)
// 	var wg sync.WaitGroup
// 	for i := 0; i < sessionCount; i++ {
// 		wg.Add(1)
// 		go func() {
// 			defer wg.Done()
// 			session := pool.GetSession()
// 			if err := warmUpSession(session); err != nil {
// 				logger.Errorf("Warmup error: %v", err)
// 			}
// 			pool.ReturnSession(session)
// 		}()
// 	}
// 	wg.Wait()
// 	logger.Info("Warmup completed")
// }

// func handleDetection(w http.ResponseWriter, r *http.Request) {
// 	if r.Method != http.MethodPost {
// 		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
// 		return
// 	}

// 	if err := r.ParseMultipartForm(32 << 20); err != nil {
// 		http.Error(w, "Failed to parse form", http.StatusBadRequest)
// 		return
// 	}

// 	results := processImages(r)

// 	w.Header().Set("Content-Type", "application/json")
// 	if err := json.NewEncoder(w).Encode(results); err != nil {
// 		logger.Errorf("Failed to encode JSON response: %v", err)
// 	}
// }

// // processImages spawns a goroutine per image and leverages the model pool for inference.
// func processImages(r *http.Request) map[string][]Detection {
// 	results := make(map[string][]Detection)
// 	var resultsMutex sync.Mutex
// 	var wg sync.WaitGroup

// 	for key, fileHeaders := range r.MultipartForm.File {
// 		if !strings.HasPrefix(key, "camera_") || len(fileHeaders) == 0 {
// 			continue
// 		}

// 		wg.Add(1)
// 		go func(key string, header *multipart.FileHeader) {
// 			defer wg.Done()

// 			file, err := header.Open()
// 			if err != nil {
// 				logger.Errorf("Error opening file %s: %v", key, err)
// 				return
// 			}
// 			defer file.Close()

// 			session := modelPool.GetSession()
// 			defer modelPool.ReturnSession(session)

// 			detections, err := detectObjects(file, session)
// 			if err != nil {
// 				logger.Errorf("Error detecting objects in %s: %v", key, err)
// 				return
// 			}

// 			resultsMutex.Lock()
// 			results[key] = detections
// 			resultsMutex.Unlock()
// 		}(key, fileHeaders[0])
// 	}

// 	wg.Wait()
// 	return results
// }

// func detectObjects(reader io.Reader, session *ModelSession) ([]Detection, error) {
// 	input, imgWidth, imgHeight, err := prepareInput(reader)
// 	if err != nil {
// 		return nil, fmt.Errorf("prepareInput error: %v", err)
// 	}

// 	output, err := runInference(input, session)
// 	if err != nil {
// 		return nil, fmt.Errorf("inference error: %v", err)
// 	}

// 	return processOutput(output, imgWidth, imgHeight), nil
// }

// func prepareInput(reader io.Reader) ([]float32, int64, int64, error) {
// 	img, _, err := image.Decode(reader)
// 	if err != nil {
// 		return nil, 0, 0, fmt.Errorf("failed to decode image: %w", err)
// 	}
// 	bounds := img.Bounds()
// 	imgWidth, imgHeight := int64(bounds.Dx()), int64(bounds.Dy())

// 	// Resize the image using Lanczos3 for quality.
// 	resized := resize.Resize(ImageSize, ImageSize, img, resize.Lanczos3)
// 	input := make([]float32, ImageSize*ImageSize*3)
// 	stride := ImageSize * ImageSize
// 	idx := 0

// 	for y := 0; y < ImageSize; y++ {
// 		for x := 0; x < ImageSize; x++ {
// 			r, g, b, _ := resized.At(x, y).RGBA()
// 			input[idx] = float32(r>>8) / 255.0
// 			input[idx+stride] = float32(g>>8) / 255.0
// 			input[idx+2*stride] = float32(b>>8) / 255.0
// 			idx++
// 		}
// 	}

// 	return input, imgWidth, imgHeight, nil
// }

// func processOutput(output []float32, imgWidth, imgHeight int64) []Detection {
//     var detections []Detection
//     var boxes []Detection
    
//     numClasses := 10 // Number of classes in yoloClasses
//     boxesPerCell := len(output) / (numClasses + 4) // 4 for x,y,w,h
    
//     // Validate output size
//     if len(output) != boxesPerCell * (numClasses + 4) {
//         logger.Errorf("Invalid output size: got %d, expected %d", 
//             len(output), boxesPerCell * (numClasses + 4))
//         return detections
//     }

//     // First pass: collect all boxes that meet confidence threshold
//     for i := 0; i < boxesPerCell; i++ {
//         // Bounds checking for box coordinates
//         if i >= len(output)/4 {
//             logger.Error("Index out of bounds when accessing box coordinates")
//             return detections
//         }

//         // Find class with highest probability
//         classID, prob := 0, float32(0.0)
//         for j := 0; j < numClasses; j++ {
//             classIndex := boxesPerCell*(j+4) + i
//             if classIndex >= len(output) {
//                 logger.Error("Index out of bounds when accessing class probabilities")
//                 return detections
//             }
            
//             if curr := output[classIndex]; curr > prob {
//                 prob = curr
//                 classID = j
//             }
//         }

//         if prob < ConfThreshold {
//             continue
//         }

//         // Get box coordinates
//         xc := output[i]
//         yc := output[boxesPerCell+i]
//         w := output[2*boxesPerCell+i]
//         h := output[3*boxesPerCell+i]

//         // Convert to image coordinates
//         x := (xc - w/2) / ImageSize * float32(imgWidth)
//         y := (yc - h/2) / ImageSize * float32(imgHeight)
//         width := w / ImageSize * float32(imgWidth)
//         height := h / ImageSize * float32(imgHeight)

//         // Validate class ID
//         if classID >= len(yoloClasses) {
//             logger.Errorf("Invalid class ID: %d", classID)
//             continue
//         }

//         detection := Detection{
//             ID:       fmt.Sprintf("detection-%d", i),
//             FromName: "object",
//             ToName:   "image",
//             Type:     "rectanglelabels",
//             Origin:   "manual",
//             Score:    prob,
//             Result: []Result{{
//                 X:      float64(x),
//                 Y:      float64(y),
//                 Width:  float64(width),
//                 Height: float64(height),
//                 Label:  yoloClasses[classID],
//             }},
//         }
//         boxes = append(boxes, detection)
//     }

//     // Sort boxes by confidence score in descending order
//     sort.Slice(boxes, func(i, j int) bool {
//         return boxes[i].Score > boxes[j].Score
//     })

//     // Apply NMS
//     selected := make([]bool, len(boxes))
//     for i := 0; i < len(boxes); i++ {
//         if selected[i] {
//             continue
//         }

//         detections = append(detections, boxes[i])
//         selected[i] = true

//         for j := i + 1; j < len(boxes); j++ {
//             if selected[j] {
//                 continue
//             }

//             // Calculate IoU between boxes[i] and boxes[j]
//             iou := calculateIoU(boxes[i].Result[0], boxes[j].Result[0])
//             if iou > IOUThreshold {
//                 selected[j] = true
//             }
//         }
//     }

//     return detections
// }

// func calculateIoU(box1, box2 Result) float32 {
// 	x1 := math.Max(box1.X, box2.X)
// 	y1 := math.Max(box1.Y, box2.Y)
// 	x2 := math.Min(box1.X+box1.Width, box2.X+box2.Width)
// 	y2 := math.Min(box1.Y+box1.Height, box2.Y+box2.Height)

// 	intersectionArea := math.Max(0, x2-x1) * math.Max(0, y2-y1)
// 	box1Area := box1.Width * box1.Height
// 	box2Area := box2.Width * box2.Height

// 	return float32(intersectionArea / (box1Area + box2Area - intersectionArea))
// }

// func runInference(input []float32, session *ModelSession) ([]float32, error) {
// 	copy(session.Input.GetData(), input)
// 	if err := session.Session.Run(); err != nil {
// 		return nil, err
// 	}
// 	return session.Output.GetData(), nil
// }

// var yoloClasses = []string{
// 	"Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
// }


package main

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"github.com/sirupsen/logrus"
)

const (
	ModelPath     = "/Users/harshnandwana/Desktop/ppe_final/yologoinference/ppe.onnx"
	ConfThreshold = 0.5
	IOUThreshold  = 0.7
	ImageSize     = 640
	// Maximum number of worker goroutines (and hence model sessions)
	MaxWorkers = 10
)

// Detection represents a single object detection result.
type Detection struct {
	ID       string   `json:"id"`
	Result   []Result `json:"result"`
	Score    float32  `json:"score"`
	FromName string   `json:"from_name"`
	ToName   string   `json:"to_name"`
	Type     string   `json:"type"`
	Origin   string   `json:"origin"`
}

// Result holds the bounding box and label.
type Result struct {
	X      float64 `json:"x"`
	Y      float64 `json:"y"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
	Label  string  `json:"label"`
}

// ModelSession holds the ONNX runtime session and its input/output tensors.
type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

// InferenceJob represents one image to process.
type InferenceJob struct {
	key        string
	fileHeader *multipart.FileHeader
	// result channel to return detection results (or error)
	resultChan chan InferenceResult
}

// InferenceResult is the result of processing one image.
type InferenceResult struct {
	key        string
	detections []Detection
	err        error
}

var (
	// jobQueue is the channel where inference jobs are sent.
	jobQueue chan InferenceJob
	// workersWg tracks our dedicated inference workers.
	workersWg sync.WaitGroup
	// logger for logging
	logger *logrus.Logger
	// yoloClasses are the class labels (modify as needed)
	yoloClasses = []string{
		"Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
		"Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
	}
)

func main() {
	logger = initLogger()
	logger.Info("Logger initialized")

	// Initialize global job queue.
	jobQueue = make(chan InferenceJob, 100)

	// Determine worker count (e.g. 80% of cores up to MaxWorkers).
	cores := runtime.NumCPU()
	workerCount := int(math.Round(float64(cores) * 0.8))
	if workerCount < 1 {
		workerCount = 1
	}
	if workerCount > MaxWorkers {
		workerCount = MaxWorkers
	}

	// Set the ONNXRuntime shared library path.
	ort.SetSharedLibraryPath(getSharedLibPath())
	if err := ort.InitializeEnvironment(); err != nil {
		logger.Fatalf("Failed to initialize ONNXRuntime environment: %v", err)
	}

	// Start dedicated worker goroutines.
	for i := 0; i < workerCount; i++ {
		session, err := createModelSession()
		if err != nil {
			logger.Fatalf("Failed to create model session %d: %v", i, err)
		}
		// Warm up the session with a dummy inference.
		if err := warmUpSession(session); err != nil {
			logger.Errorf("Warmup error for session %d: %v", i, err)
		}
		workersWg.Add(1)
		go inferenceWorker(i, session)
	}
	logger.Infof("Started %d inference workers", workerCount)

	http.HandleFunc("/detect", handleDetection)
	logger.Info("Server starting on :8000")
	if err := http.ListenAndServe(":8000", nil); err != nil {
		logger.Fatalf("Server failed: %v", err)
	}
	// When shutting down, close the jobQueue and wait for workers:
	// close(jobQueue)
	// workersWg.Wait()
}

// inferenceWorker is a dedicated worker that processes incoming jobs.
// It locks its OS thread to reserve a core for inference.
func inferenceWorker(workerID int, session *ModelSession) {
	// Lock this goroutine to its OS thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer workersWg.Done()

	logger.Infof("Worker %d started", workerID)

	for job := range jobQueue {
		// Open the file from the header.
		file, err := job.fileHeader.Open()
		if err != nil {
			logger.Errorf("Worker %d: failed to open file %s: %v", workerID, job.key, err)
			job.resultChan <- InferenceResult{key: job.key, err: err}
			continue
		}

		detections, err := detectObjects(file, session)
		file.Close() // ensure file is closed after processing

		job.resultChan <- InferenceResult{key: job.key, detections: detections, err: err}
	}
	logger.Infof("Worker %d exiting", workerID)
}

// handleDetection is the HTTP handler that receives multipart form images.
func handleDetection(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse multipart form with a 32MB limit.
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	// For each file with key starting with "camera_", create a job.
	var wg sync.WaitGroup
	resultsMutex := sync.Mutex{}
	results := make(map[string][]Detection)

	for key, fileHeaders := range r.MultipartForm.File {
		if !strings.HasPrefix(key, "camera_") || len(fileHeaders) == 0 {
			continue
		}
		wg.Add(1)
		go func(key string, fh *multipart.FileHeader) {
			defer wg.Done()
			// Create a job with its own result channel.
			job := InferenceJob{
				key:        key,
				fileHeader: fh,
				resultChan: make(chan InferenceResult, 1),
			}
			// Send the job to the global jobQueue.
			jobQueue <- job
			// Wait for the result.
			res := <-job.resultChan
			if res.err != nil {
				logger.Errorf("Error processing %s: %v", key, res.err)
				return
			}
			resultsMutex.Lock()
			results[key] = res.detections
			resultsMutex.Unlock()
		}(key, fileHeaders[0])
	}
	wg.Wait()

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(results); err != nil {
		logger.Errorf("Failed to encode JSON response: %v", err)
	}
}

// detectObjects decodes an image, prepares input, runs inference, and processes output.
func detectObjects(reader io.Reader, session *ModelSession) ([]Detection, error) {
	input, imgWidth, imgHeight, err := prepareInput(reader)
	if err != nil {
		return nil, fmt.Errorf("prepareInput error: %v", err)
	}

	output, err := runInference(input, session)
	if err != nil {
		return nil, fmt.Errorf("inference error: %v", err)
	}

	return processOutput(output, imgWidth, imgHeight), nil
}

// prepareInput decodes, resizes, and normalizes an image.
func prepareInput(reader io.Reader) ([]float32, int64, int64, error) {
	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to decode image: %w", err)
	}
	bounds := img.Bounds()
	imgWidth, imgHeight := int64(bounds.Dx()), int64(bounds.Dy())

	resized := resize.Resize(ImageSize, ImageSize, img, resize.Lanczos3)
	input := make([]float32, ImageSize*ImageSize*3)
	stride := ImageSize * ImageSize
	idx := 0

	for y := 0; y < ImageSize; y++ {
		for x := 0; x < ImageSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			input[idx] = float32(r>>8) / 255.0
			input[idx+stride] = float32(g>>8) / 255.0
			input[idx+2*stride] = float32(b>>8) / 255.0
			idx++
		}
	}
	return input, imgWidth, imgHeight, nil
}

// processOutput processes the model output into detections.
func processOutput(output []float32, imgWidth, imgHeight int64) []Detection {
	var detections []Detection
	var boxes []Detection

	numClasses := 10 // Number of classes in yoloClasses
	boxesPerCell := len(output) / (numClasses + 4) // 4 for x,y,w,h

	if len(output) != boxesPerCell*(numClasses+4) {
		logger.Errorf("Invalid output size: got %d, expected %d", len(output), boxesPerCell*(numClasses+4))
		return detections
	}

	for i := 0; i < boxesPerCell; i++ {
		// Find class with highest probability.
		classID, prob := 0, float32(0.0)
		for j := 0; j < numClasses; j++ {
			classIndex := boxesPerCell*(j+4) + i
			if classIndex >= len(output) {
				logger.Error("Index out of bounds accessing class probabilities")
				return detections
			}
			if curr := output[classIndex]; curr > prob {
				prob = curr
				classID = j
			}
		}
		if prob < ConfThreshold {
			continue
		}

		xc := output[i]
		yc := output[boxesPerCell+i]
		w := output[2*boxesPerCell+i]
		h := output[3*boxesPerCell+i]

		x := (xc - w/2) / ImageSize * float32(imgWidth)
		y := (yc - h/2) / ImageSize * float32(imgHeight)
		width := w / ImageSize * float32(imgWidth)
		height := h / ImageSize * float32(imgHeight)

		if classID >= len(yoloClasses) {
			logger.Errorf("Invalid class ID: %d", classID)
			continue
		}

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

	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Score > boxes[j].Score
	})

	// Non-Maximum Suppression (NMS)
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
			iou := calculateIoU(boxes[i].Result[0], boxes[j].Result[0])
			if iou > IOUThreshold {
				selected[j] = true
			}
		}
	}

	return detections
}

// calculateIoU computes the Intersection-over-Union of two boxes.
func calculateIoU(box1, box2 Result) float32 {
	x1 := math.Max(box1.X, box2.X)
	y1 := math.Max(box1.Y, box2.Y)
	x2 := math.Min(box1.X+box1.Width, box2.X+box2.Width)
	y2 := math.Min(box1.Y+box1.Height, box2.Y+box2.Height)

	intersectionArea := math.Max(0, x2-x1) * math.Max(0, y2-y1)
	box1Area := box1.Width * box1.Height
	box2Area := box2.Width * box2.Height

	return float32(intersectionArea / (box1Area + box2Area - intersectionArea))
}

// runInference copies the input data to the session and runs the model.
func runInference(input []float32, session *ModelSession) ([]float32, error) {
	copy(session.Input.GetData(), input)
	if err := session.Session.Run(); err != nil {
		return nil, err
	}
	return session.Output.GetData(), nil
}

// createModelSession creates a new ONNX runtime session for the model.
func createModelSession() (*ModelSession, error) {
	inputShape := ort.NewShape(1, 3, ImageSize, ImageSize)
	inputTensor, err := ort.NewTensor(inputShape, make([]float32, ImageSize*ImageSize*3))
	if err != nil {
		return nil, err
	}

	outputShape := ort.NewShape(1, 14, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()

	// Restrict threads per session.
	options.SetIntraOpNumThreads(1)
	options.SetInterOpNumThreads(1)

	session, err := ort.NewAdvancedSession(
		ModelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		options,
	)
	if err != nil {
		return nil, err
	}

	return &ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}, nil
}

// warmUpSession runs a dummy inference to ensure the session is fully loaded.
func warmUpSession(session *ModelSession) error {
	dummyInput := make([]float32, ImageSize*ImageSize*3)
	copy(session.Input.GetData(), dummyInput)
	return session.Session.Run()
}

// getSharedLibPath returns the ONNXRuntime shared library path based on OS.
func getSharedLibPath() string {
	switch runtime.GOOS {
	case "windows":
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll"
		}
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib"
		}
	case "linux":
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so"
		}
		return "./third_party/onnxruntime.so"
	}
	panic("Unable to find a version of the ONNXRuntime library supporting this system.")
}

// initLogger initializes a Logrus logger that outputs to both stdout and a file.
func initLogger() *logrus.Logger {
	log := logrus.New()
	log.SetFormatter(&logrus.TextFormatter{FullTimestamp: true})
	file, err := os.OpenFile("app.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Warn("Failed to log to file, using default stderr")
	} else {
		log.SetOutput(io.MultiWriter(os.Stdout, file))
	}
	return log
}