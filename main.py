from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from PIL import Image
import os
import random

load_dotenv()
hf_token = os.getenv("HF_HUB_API_KEY")

prompt = input("Enter a prompt for the image generation: ")
random_num = random.randrange(1, 1000)

client = InferenceClient(
    provider="replicate",
    api_key=hf_token,
)

print("Starting inference...")

image = client.text_to_image(
    prompt=prompt, 
    model="black-forest-labs/FLUX.1-dev"
)

image.save(f"output_{random_num}.png")
print(f"Inference completed. Image saved as output_{random_num}.png")