try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:  # ImportError or other issues when tensorflow isn't present
    tf = None
    TF_AVAILABLE = False
import numpy as np
from PIL import Image
import io
import os

# Global variable to hold the loaded model
MODEL = None
# Define target image size based on your model's input requirements (MNIST is 28x28 grayscale)
IMAGE_SIZE = (28, 28)  # Adjusted for MNIST
CLASS_LABELS = [str(i) for i in range(10)]  # MNIST digits 0-9

def load_model(model_path: str = None):
    global MODEL
    if MODEL is None:
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed or failed to import.")
        # Default model path, can be overridden by environment variable
        effective_model_path = model_path if model_path else os.environ.get("MODEL_PATH", "models/my_classifier_model.h5")
        if not os.path.exists(effective_model_path):
            raise FileNotFoundError(f"Model file not found at {effective_model_path}")
        MODEL = tf.keras.models.load_model(effective_model_path)
        # Compile the model for consistency
        MODEL.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return MODEL

def is_model_loaded() -> bool:
    """Returns True if the ML model is currently loaded in memory."""
    return MODEL is not None

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # Decode image from bytes, convert to grayscale, resize, and normalize pixel values.
    if not image_bytes:
        raise ValueError("Empty image file payload.")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image_array = np.expand_dims(image_array, axis=[0, -1])  # Add batch and channel dimensions (batch_size, height, width, channels)
        return image_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def predict_image(preprocessed_image: np.ndarray):
    model = load_model()  # Ensure model is loaded
    predictions = model.predict(preprocessed_image)
    # Convert raw predictions (e.g., softmax outputs) into meaningful class labels and probabilities.
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    probabilities = predictions[0].tolist()  # Convert numpy array to list for JSON serialization

    return {
        "class_label": CLASS_LABELS[predicted_class_idx],
        "probabilities": probabilities
    }

