# ---- Step 1: Build Binary ----
FROM --platform=linux/amd64 golang:1.20 as builder
# Set environment variables for cross-compilation and CGO
ENV GOOS=linux \
    GOARCH=amd64 \
    CGO_ENABLED=1

# Set working directory inside the container
WORKDIR /app

# Install necessary dependencies for cross-compilation
RUN apt update && apt install -y wget tar gcc g++

# Initialize a new Go module if go.mod doesn't exist
RUN go mod init ppe_detector

# Add required dependencies explicitly
RUN go get github.com/nfnt/resize && \
    go get github.com/sirupsen/logrus && \
    go get github.com/yalue/onnxruntime_go

# Copy source files
COPY go.mod go.sum ./
COPY main.go ./
RUN go mod tidy
COPY ppe.onnx ./
COPY third_party ./third_party

# Download ONNX Runtime (for x86_64)
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz && \
    tar -xvzf onnxruntime-linux-x64-1.15.1.tgz && \
    mv onnxruntime-linux-x64-1.15.1 /usr/local/onnxruntime

# Set CGO flags to correctly link ONNX Runtime
ENV CGO_CFLAGS="-I/usr/local/onnxruntime/include"
ENV CGO_LDFLAGS="-L/usr/local/onnxruntime/lib -lonnxruntime"

# Build the Go binary
RUN go build -o ppe_detector main.go

# ---- Step 2: Extract Binary to /output ----
WORKDIR /output
RUN cp /app/ppe_detector /output/ppe_detector