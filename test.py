import os
import json
from gravixlayer import GravixLayer
from dotenv import load_dotenv

load_dotenv()
# export GRAVIXLAYER_API_KEY=your_api_key_here
client = GravixLayer()

embedding = client.embeddings.create(
    model="nomic-ai/nomic-embed-text:v1.5",
    input="Why is the sky blue?",
)

print(json.dumps(embedding.model_dump(), indent=2))