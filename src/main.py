from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
from contextlib import asynccontextmanager
from src.model import load_model, preprocess_image, predict_image, is_model_loaded
import os

# Configure basic logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if os.environ.get("SKIP_MODEL_LOAD") != "1":
        try:
            # The model path is configurable via an environment variable for flexibility
            model_path_env = os.environ.get("MODEL_PATH")
            if model_path_env:
                logger.info(f"Attempting to load model from ENV: {model_path_env}")
                load_model(model_path=model_path_env)
            else:
                logger.info("Attempting to load model from default path: models/my_classifier_model.h5")
                load_model()
            logger.info("ML Model loaded successfully and ready for inference.")
        except FileNotFoundError as e:
            logger.critical(f"Critical: Model file not found. Application cannot start: {e}")
            raise RuntimeError(f"Model not found at specified path. Check MODEL_PATH environment variable or default path.")
        except Exception as e:
            logger.critical(f"Critical: Failed to load ML Model due to an unexpected error: {e}")
            raise RuntimeError(f"Failed to load ML Model: {e}")
    yield
    # Shutdown (if needed)
    pass

app = FastAPI(title="ML Prediction API", description="API for image classification using Keras model.", lifespan=lifespan)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the response model for consistent API output
class PredictionResponse(BaseModel):
    class_label: str
    probabilities: List[float]

# Health check endpoint for monitoring application status
@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    return {"status": "ok", "message": "API is healthy and model is loaded."}

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the ML API. Visit /docs for API documentation.", "docs_url": "/docs"}

# Prediction endpoint accepting an image file
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, summary="Get Image Prediction")
async def predict(file: UploadFile = File(..., description="Image file to classify")):
    # Basic input validation for content type
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Received invalid file type: {file.content_type}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only image files (e.g., JPEG, PNG) are allowed for prediction.")
    
    try:
        # Read image bytes, preprocess, and make a prediction
        image_bytes = await file.read()
        preprocessed_image = preprocess_image(image_bytes)
        prediction_result = predict_image(preprocessed_image)
        logger.info(f"Prediction successful for {file.filename}: {prediction_result['class_label']}")
        return PredictionResponse(**prediction_result)
    except ValueError as ve:
        # Handle specific data processing errors
        logger.error(f"Data preprocessing error for {file.filename}: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Image processing failed: {ve}")
    except Exception as e:
        # Catch any other unexpected errors during prediction
        logger.error(f"Unexpected prediction failure for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal server error occurred during prediction: {e}")
