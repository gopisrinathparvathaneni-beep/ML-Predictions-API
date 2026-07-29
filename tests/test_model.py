import pytest
import numpy as np
import io
from PIL import Image
import os
import sys

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import preprocess_image, IMAGE_SIZE, CLASS_LABELS

def test_preprocess_valid_image():
    # Create a 50x50 RGB image in memory
    img = Image.new('RGB', (50, 50), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img_bytes = buf.getvalue()

    processed = preprocess_image(img_bytes)

    # Check shape: (1, 28, 28, 1)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    # Check normalized pixel values between 0 and 1
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0

def test_preprocess_empty_bytes():
    with pytest.raises(ValueError, match="Empty image file payload"):
        preprocess_image(b"")

def test_preprocess_invalid_bytes():
    with pytest.raises(ValueError, match="Error processing image"):
        preprocess_image(b"not an image file data stream")

def test_class_labels_length():
    assert len(CLASS_LABELS) == 10
    assert CLASS_LABELS[0] == "0"
    assert CLASS_LABELS[9] == "9"
