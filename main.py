from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from PIL import Image
import os

load_dotenv()
hf_token = os.getenv("HF_HUB_API_KEY")

client = InferenceClient(
    provider="replicate",
    api_key=hf_token,
)

print("Starting inference...")

image = client.text_to_image(
    "An astronaut riding a horse on Mars", 
    model="black-forest-labs/FLUX.1-dev"
)

image.save("output.png")