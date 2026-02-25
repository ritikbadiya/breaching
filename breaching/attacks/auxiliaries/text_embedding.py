from transformers import CLIPTokenizer, BertTokenizer, BertModel, CLIPTextModel
import torch
from class_names import cifar10_classes, imagenet_classes
import torch.nn.functional as F

def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    mean_col = gram.mean(dim=0, keepdim=True)
    mean_row = gram.mean(dim=1, keepdim=True)
    mean_all = gram.mean()
    return gram - mean_col - mean_row + mean_all

def centered_kernel_alignment(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute linear CKA between two embedding matrices of shape [n, d]."""
    if embeddings_a.ndim == 1:
        embeddings_a = embeddings_a.unsqueeze(0)
    if embeddings_b.ndim == 1:
        embeddings_b = embeddings_b.unsqueeze(0)
    if embeddings_a.shape[0] != embeddings_b.shape[0]:
        raise ValueError("CKA requires the same number of samples in both embeddings.")
    if embeddings_a.shape[0] < 2:
        raise ValueError("CKA requires at least two samples (n >= 2).")

    embeddings_a = embeddings_a.float()
    embeddings_b = embeddings_b.float()

    gram_a = embeddings_a @ embeddings_a.T
    gram_b = embeddings_b @ embeddings_b.T
    gram_a = _center_gram(gram_a)
    gram_b = _center_gram(gram_b)

    hsic = (gram_a * gram_b).sum()
    denom = torch.norm(gram_a) * torch.norm(gram_b)
    return hsic / (denom + eps)

class TextEmbedder:
    def __init__(self, model_name="bert-base-uncased", device="cuda"):
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.model = BertModel.from_pretrained(model_name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.device = device

    def embed(self, text):
        inputs = self.tokenizer(text, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=self.tokenizer.model_max_length
                                ).to(self.device) # 'pt' indicates that hte tokenizer should return PyTorch tensors
        # print(inputs)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output  # Return the mean embedding of the sequence ==> last_hidden_state[:, 0, :]
    
@torch.no_grad()
def encode_classes(classnames, templates, embedder, device="cuda"):
    all_embeddings = []
    
    for name in classnames:
        texts = [t.format(name.replace("_", " ")) for t in templates]
        embeds = embedder.embed(texts) # (T, D)
        embeds = F.normalize(embeds, dim=-1)
        mean_embed = embeds.mean(dim=0)
        mean_embed = F.normalize(mean_embed, dim=-1)
        all_embeddings.append(mean_embed)

    return torch.stack(all_embeddings, dim=0)  # (num_classes, D)

if __name__ == "__main__":
    embedder = TextEmbedder()

    templates = [
        "a photo of a {}",
        "an image of a {}",
        # "a blurry photo of a {}",
        # "a black and white photo of a {}",
        "a cropped photo of a {}",
        "a close-up photo of a {}",
        # "a bright photo of a {}",
    ]

    cifar_embeds = encode_classes(cifar10_classes, templates, embedder)
    imagenet_embeds = encode_classes(imagenet_classes, templates, embedder)
    
    similarity = cifar_embeds @ imagenet_embeds.T  # (Ncifar, Nimagenet)
    topk = 5
    values, indices = similarity.topk(topk, dim=-1)

    for i, cifar_name in enumerate(cifar10_classes):
        print(f"\nCIFAR class: {cifar_name}")
        for rank in range(topk):
            idx = indices[i, rank].item()
            print(f"  {rank+1}. {imagenet_classes[idx]} (sim={values[i, rank].item():.4f})")