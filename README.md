# ML Image Classification API

A production-ready RESTful API for image classification using a pre-trained Keras model, containerized with Docker and automated with GitHub Actions CI/CD.

Repository: `https://github.com/gopisrinathparvathaneni-beep/ML-Predictions-API.git`

## Features

- **RESTful API**: Built with FastAPI for high-performance image classification predictions.
- **Image Classification**: Supports MNIST digit classification (0-9) with probability outputs.
- **Input Validation**: Robust validation for image file uploads (JPEG, PNG).
- **Error Handling**: Comprehensive error handling with appropriate HTTP status codes.
- **Health Check**: `GET /health` endpoint for monitoring API and model status.
- **CORS Support**: Cross-Origin Resource Sharing enabled for web app integration.
- **Structured Logging**: Configurable logging for debugging and monitoring.
- **Docker Containerization**: Optimized multi-stage Dockerfile for efficient deployment.
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing and deployment.
- **Environment Configuration**: Configurable via environment variables.

## Technology Stack

- **Python 3.9+**
- **FastAPI**: Web framework for API development
- **TensorFlow/Keras**: Deep learning framework for model inference
- **Uvicorn**: ASGI server for FastAPI
- **Pillow**: Image processing library
- **Docker**: Containerization platform
- **GitHub Actions**: CI/CD automation
- **pytest**: Testing framework

## Project Structure

```
ML-Predictions-API/
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI application, CORS middleware, and endpoints
│   └── model.py         # Model loading, status check, preprocessing & inference logic
├── models/
│   └── my_classifier_model.h5  # Pre-trained Keras model
├── tests/
│   ├── test_api.py      # Unit tests for API endpoints (/health, /predict, /)
│   └── test_model.py    # Unit tests for image preprocessing and label definitions
├── predictions/
│   ├── example_prediction_0.json
│   └── example_prediction_9.json  # Example prediction outputs
├── .github/workflows/
│   └── main.yml         # GitHub Actions CI/CD pipeline
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Local development setup
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── train_model.py       # Script to train and save the MNIST model
└── README.md            # Project documentation and usage guide
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git

### Local Development

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ml-api
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional - pre-trained model included):**
   ```bash
   python train_model.py
   ```

4. **Run tests:**
   ```bash
   pytest tests/
   ```

5. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

The API will be available at `http://localhost:8000`.

## API Usage

### Health Check

**GET /health**

Check if the API and model are ready.

**Response:**
```json
{
  "status": "ok",
  "message": "API is healthy and model is loaded."
}
```

### Image Prediction

**POST /predict**

Classify an uploaded image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file - JPEG, PNG)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.png"
```

**Response:**
```json
{
  "class_label": "5",
  "probabilities": [0.05, 0.05, 0.05, 0.05, 0.05, 0.75, 0.0, 0.0, 0.0, 0.0]
}
```

### Error Responses

**400 Bad Request** - Invalid file type:
```json
{
  "detail": "Only image files (e.g., JPEG, PNG) are allowed for prediction."
}
```

**422 Unprocessable Entity** - Image processing error:
```json
{
  "detail": "Image processing failed: [error message]"
}
```

**500 Internal Server Error** - Server error:
```json
{
  "detail": "An internal server error occurred during prediction: [error message]"
}
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Tests include:
- Health check endpoint validation
- Prediction endpoint with mocked model
- Input validation for file types
- Error handling scenarios

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/main.yml`) automates:

1. **Checkout code** on push to main branch
2. **Set up Python environment**
3. **Install dependencies**
4. **Run unit tests**
5. **Build Docker image**
6. **Simulate deployment** (placeholder for actual registry push)

### Workflow Triggers

- Push to `main` branch
- Pull requests to `main` branch

### Viewing Pipeline Status

Check the "Actions" tab in your GitHub repository to monitor workflow runs.

## Deployment

### Local Deployment

```bash
docker-compose up --build
```

### Production Deployment

1. Build the Docker image:
   ```bash
   docker build -t ml-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 -e MODEL_PATH=/app/models/my_classifier_model.h5 ml-api
   ```

### Environment Variables

Configure the following environment variables:

- `MODEL_PATH`: Path to the Keras model file (default: `models/my_classifier_model.h5`)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)

## Prediction Examples

See the `predictions/` directory for example JSON outputs:

- `example_prediction_0.json`: Prediction for digit 0
- `example_prediction_9.json`: Prediction for digit 9

## Future Enhancements

- Add authentication and authorization
- Implement model versioning and A/B testing
- Add monitoring and metrics collection
- Support for multiple model formats
- Batch prediction endpoints
- Model retraining pipeline
- Advanced CI/CD with actual registry deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
