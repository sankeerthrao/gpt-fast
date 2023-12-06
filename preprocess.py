import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class MathBookDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Load your book text here (after converting to plain text)
book_texts = ["Segment 1 text", "Segment 2 text", ...]  # Replace with actual segments

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Or any suitable tokenizer
dataset = MathBookDataset(book_texts, tokenizer, max_len=2048)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
