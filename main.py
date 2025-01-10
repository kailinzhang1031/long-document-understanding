
# Load dataset

import json
from torch.utils.data import Dataset

import torch

def maxsim_score_torch(Eq: torch.Tensor, Ep: torch.Tensor) -> torch.Tensor:
    """
    Computes the MaxSim score between the query embedding Eq and page embedding Ep 
    using PyTorch tensors.

    Parameters
    ----------
    Eq : torch.Tensor
        A tensor of shape (nq, d), where nq is the number of query tokens, 
        and d is the embedding dimension. 
        Each row is a d-dimensional embedding vector for the query.
    Ep : torch.Tensor
        A tensor of shape (nv, d), where nv is the number of page tokens, 
        and d is the embedding dimension. 
        Each row is a d-dimensional embedding vector for the page.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the MaxSim score, s(q, p):
        
           s(q, p) = sum_{i=1 to nq} max_{j in [nv]} dot(Eq[i], Ep[j]).
    """
    # 1. Compute dot-product matrix M of shape (nq, nv),
    #    where M[i, j] = Eq[i] dot Ep[j].
    M = Eq @ Ep.t()  # (nq, nv)

    # 2. For each query token i, find the maximum dot-product across all page tokens j.
    rowwise_max = torch.max(M, dim=1).values  # shape: (nq,)

    # 3. Sum over the query dimension to get the final score.
    return rowwise_max.sum()

def retrieve_top_k_pages(Eq: torch.Tensor,
                         Ec: torch.Tensor,
                         k: int):
    """
    Retrieve top-k pages from a collection Ec, based on the MaxSim score 
    with a query embedding Eq.

    Parameters
    ----------
    Eq : torch.Tensor
        Shape: (nq, d)
        A single query embedding matrix with nq query tokens.
    Ec : torch.Tensor
        Shape: (N, nv, d)
        A collection of N pages, each page embedding has nv tokens 
        (rows) in d dimensions (columns).
    k : int
        Number of pages to retrieve.

    Returns
    -------
    topk_scores : torch.Tensor
        Shape: (k,)
        The top-k MaxSim scores in descending order.
    topk_indices : torch.Tensor
        Shape: (k,)
        The indices of the top-k pages (0-based).
    topk_pages : torch.Tensor
        Shape: (k, nv, d)
        The actual embeddings of the top-k pages in Ec.
    """
    # ------------------------------------------------------------
    # 1) Compute MaxSim for each page, in a batched/vectorized way.
    #
    # We want M[i] = Eq @ Ec[i].T for each i in [0..N-1],
    #   which yields shape (nq, nv).
    # Then, for each row i, we do max over columns j and sum over i.
    #
    # Vectorized approach:
    #   - Eq has shape (nq, d).
    #   - Ec has shape (N, nv, d).
    #   - We'll create batch versions so we can do a single bmm() call:
    #       M = bmm( Eq_batch, Ec_batch^T )
    #     resulting in shape (N, nq, nv).
    # ------------------------------------------------------------

    N, nv, d = Ec.shape
    nq = Eq.shape[0]

    # Expand Eq to shape (N, nq, d) so it can broadcast in a batch of size N
    Eq_batch = Eq.repeat(N, 1, 1)  # shape: (N, nq, d)

    # Transpose Ec to shape (N, d, nv) so we can do batch-matmul
    Ec_transpose = Ec.transpose(1, 2)            # shape: (N, d, nv)

    # Batch-matrix-multiply:
    #   Eq_batch : (N, nq, d)
    #   Ec_transpose : (N, d, nv)
    # => M : (N, nq, nv) where M[i] = Eq[i] @ Ec[i].T
    M = torch.bmm(Eq_batch, Ec_transpose)  # shape: (N, nq, nv)

    # For each page i: M[i] is size (nq, nv).
    # We take max over dim=2 (the "nv" dimension), which yields shape (N, nq).
    # Then sum over dim=1 (the "nq" dimension) => final shape (N,)
    rowwise_max = torch.max(M, dim=2).values     # shape: (N, nq)
    scores = rowwise_max.sum(dim=1)             # shape: (N,)

    # ------------------------------------------------------------
    # 2) Extract top-k pages by MaxSim score
    # ------------------------------------------------------------
    topk_scores, topk_indices = torch.topk(scores, k, largest=True, sorted=True)
    # Grab the pages from Ec using these indices
    topk_pages = Ec[topk_indices]

    return topk_scores, topk_indices, topk_pages


class QADataset(Dataset):
    """
    A custom Dataset for QA (Question-Answer) data stored in JSON format.

    The JSON file should contain a list of objects,
    each having at least the following structure:
    {
      "doc_id": str or int,
      "question": str,
      "answer": str
    }
    """

    def __init__(self, json_file_path):
        """
        :param json_file_path: Path to the JSON file containing the dataset.
        :param transform: Optional transform to apply to the data (e.g., tokenization).
        """
        super().__init__()
        self.json_file_path = json_file_path
        self.data = self._load_data()

    def _load_data(self):
        """
        Load and parse the JSON file into a Python list of items.
        Each item must contain keys: 'doc_id', 'question', 'answer'.
        """
        with open(self.json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def __len__(self):
        """
        :return: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx: Index for the sample to retrieve.
        :return: A single sample as a dict or a tuple, depending on your preference.
        """
        sample = self.data[idx]
        doc_id = sample["doc_id"]
        question = sample["question"]
        answer = sample["answer"]


        # Return the sample in the desired format
        return {
            "doc_id": doc_id,
            "question": question,
            "answer": answer
        }

# Example usage:
# dataset_path = "/root/autodl-tmp/data/samples.json"
# dataset = QADataset(dataset_path)

# print(f"Dataset size: {len(dataset)}")
# first_sample = dataset[0]
# print(first_sample)

# Load the model

model_name = "/root/autodl-tmp/model/vidore/colpali-v1.3"

import torch
import pickle
from PIL import Image

from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device

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

# Load the image and query
# image = Image.open("shift_kazakhstan.jpg")

image_path = "/root/autodl-tmp/data/documents_np/2005.12872v3.np"

with open(image_path,"rb") as f:
    pages = pickle.load(f)

images = [Image.fromarray(page) for page in pages]

query = "How many multi-head self-attention layers does DETR's default structure have?"

# Preprocess inputs
batch_images = [processor.process_images([image]).to(device) for image in images]
batch_queries = processor.process_queries([query]).to(device)


# Forward passes
with torch.no_grad():
    image_embeddings = [model.forward(**batch_image) for batch_image in batch_images]
    query_embeddings = model.forward(**batch_queries)

image_embeddings = torch.cat(image_embeddings,dim=0)
topk_scores, topk_indices, topk_pages = retrieve_top_k_pages(query_embeddings,image_embeddings,k=2)


# # Get the number of image patches
# n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size)

# # Get the tensor mask to filter out the embeddings that are not related to the image
# image_mask = processor.get_image_mask(batch_images)

# # Generate the similarity maps
# batched_similarity_maps = get_similarity_maps_from_embeddings(
#     image_embeddings=image_embeddings,
#     query_embeddings=query_embeddings,
#     n_patches=n_patches,
#     image_mask=image_mask,
# )

# # Get the similarity map for our (only) input image
# similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)

# # Tokenize the query
# query_tokens = processor.tokenizer.tokenize(query)

# # Plot and save the similarity maps for each query token
# plots = plot_all_similarity_maps(
#     image=image,
#     query_tokens=query_tokens,
#     similarity_maps=similarity_maps,
# )
# for idx, (fig, ax) in enumerate(plots):
#     fig.savefig(f"similarity_map_{idx}.png")