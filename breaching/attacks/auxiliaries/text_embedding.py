from transformers import CLIPTokenizer, BertTokenizer, BertModel
import torch

class TextEmbedder:
    def __init__(self, model_name="bert-base-uncased", device="cuda"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.device = device

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device) # 'pt' indicates that hte tokenizer should return PyTorch tensors
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Return the mean embedding of the sequence


if __name__ == "__main__":
    embedder = TextEmbedder()
    text = "This is a test sentence."
    embedding = embedder.embed(text)
    print(embedding)