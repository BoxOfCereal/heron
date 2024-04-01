import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from heron.datasets.smw_agent_dataset import SMWAgentDataset
from heron.datasets.vg_datasets import load_image 

d = SMWAgentDataset.create(
    
)

# Load from URL
image_url = "https://example.com/image.jpg"
loaded_image_url = load_image(image_url)

# Load from file path
file_path = "path/to/image.jpg"
loaded_image_file = load_image(file_path)

# Load from base64 string
base64_string = "base64_string_here"
loaded_image_base64 = load_image(base64.b64decode(base64_string))

# Load from PIL Image object
pil_image = Image.open("path/to/image.jpg")
loaded_image_pil = load_image(pil_image)

# Load from numpy array (tensor)
image_tensor = np.random.rand(224, 224, 3)  # Example tensor with shape (height, width, channels)
loaded_image_tensor = load_image(image_tensor)
