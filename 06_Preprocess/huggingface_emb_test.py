from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = ["한글", "언어", "영어", "한국어", "english"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings_np = sentence_embeddings.numpy()

A = embeddings_np[0].reshape(1, -1)
B = embeddings_np[1].reshape(1, -1)
C = embeddings_np[2].reshape(1, -1)
D = embeddings_np[3].reshape(1, -1)
E = embeddings_np[4].reshape(1, -1)

li = ["A", "B", "C", "D", "E"]

from itertools import combinations

for i, j in combinations(li, 2):
    idx_i = li.index(i)
    idx_j = li.index(j)
    similarity = cosine_similarity(embeddings_np[idx_i].reshape(1, -1), embeddings_np[idx_j].reshape(1, -1))
    print(f"'{sentences[idx_i]}'와(과) '{sentences[idx_j]}'의 코사인 유사도: {similarity[0][0]:.4f}")