import json
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

import os
import io


import pickle
import pyarrow.parquet as pq
import faiss

from tqdm import tqdm
from PIL import Image

from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device


# Retrieval function

# see http://ulrichpaquet.com/Papers/SpeedUp.pdf theorem 5

def get_phi(xb): 
    return (xb ** 2).sum(1).max()

def augment_xb(xb, phi=None): 
    norms = (xb ** 2).sum(1)
    if phi is None: 
        phi = norms.max()
    extracol = np.sqrt(phi - norms)
    return np.hstack((xb, extracol.reshape(-1, 1)))

def augment_xq(xq): 
    extracol = np.zeros(len(xq), dtype='float32')
    return np.hstack((xq, extracol.reshape(-1, 1)))

# Load model

# model_name = "/root/autodl-tmp/model/vidore/colpali-v1.3"

model_name = "vidore/colpali-v1.3"
device = get_torch_device("auto")

# Load the model
model = ColPali.from_pretrained(
    model_name,
    # torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

# Load the processor
processor = ColPaliProcessor.from_pretrained(model_name)

# Load dataset

images_dir_path = "/root/autodl-tmp/data_1/lmms-lab/MP-DocVQA/data"
test_images_path = os.path.join(images_dir_path,"test_images.pkl")
test_images_embedding_path = os.path.join(images_dir_path,"test_images_embedding.pkl")
test_queries_embedding_path = os.path.join(images_dir_path,"test_queries_embedding.pkl")

query_list = []
with open(test_images_path,"rb") as d:
    images_base = pickle.load(d)



image_embeddings = []

for image_dict in tqdm(images_base):
    # Read the image from bytes
    image_bytes = image_dict["bytes"]
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img = pil_img.resize([1700,2200])    
    # Convert the image to embedding
    image = processor.process_images([pil_img]).to(device)
    img_embedding = model.forward(**image)  # shape: (d,)

    # Store (image embedding, path) for later
    image_embeddings.append(img_embedding)

image_embeddings = torch.concat(image_embeddings)

with open(test_images_embedding_path,"wb") as im_embed:
    pickle.dump(image_embeddings,im_embed)

query_embeddings = []
for query in tqdm(query_list):
    query = processor.process_queries(query)
    query_embedding = model.forward(**query)
    query_embeddings.append(query_embedding)

query_embeddings = torch.concat(query_embeddings)

with open(test_queries_embedding_path,"wb") as qr_embed:
    pickle.dump(query_embeddings,qr_embed)

d = query_embedding.shape[-1]

# reference IP search
k = 10
index = faiss.IndexFlatL2(d + 1)
index.add(augment_xb(xb=image_embeddings))
D, I = index.search(augment_xq(xq=query_embeddings), k)



# Process image


# Retrieval