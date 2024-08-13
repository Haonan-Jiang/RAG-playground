import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np

# Decision: Choose base model
BASE_MODEL = 'bert-base-uncased'  # Could be RoBERTa, SentenceTransformer, etc.

class EmbeddingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Decision: How to create pairs/triplets
        text1, text2 = self.texts[idx]
        label = self.labels[idx]

        encoding1 = self.tokenizer.encode_plus(
            text1,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoding2 = self.tokenizer.encode_plus(
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids1': encoding1['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class EmbeddingModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Decision: Add task-specific layers?
        self.pooler = torch.nn.Linear(768, 768)  # Example of added layer

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        # Decision: How to pool embeddings
        pooled_output = self.pooler(outputs.last_hidden_state[:, 0, :])
        return pooled_output

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    # Decision: Choose loss function
    criterion = torch.nn.CosineEmbeddingLoss()
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            emb1 = model(batch['input_ids1'].to(device), batch['attention_mask1'].to(device))
            emb2 = model(batch['input_ids2'].to(device), batch['attention_mask2'].to(device))
            labels = batch['label'].to(device)
            
            loss = criterion(emb1, emb2, labels)
            loss.backward()
            
            # Decision: Gradient clipping?
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                emb1 = model(batch['input_ids1'].to(device), batch['attention_mask1'].to(device))
                emb2 = model(batch['input_ids2'].to(device), batch['attention_mask2'].to(device))
                labels = batch['label'].to(device)
                val_loss += criterion(emb1, emb2, labels).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader)}")
        
        # Decision: Implement early stopping?

# Main process
def main():
    # Load data
    # Decision: How much data? Data augmentation?
    texts = [("This is a positive pair", "This is similar"), 
             ("This is a negative pair", "This is different")]
    labels = [1, 0]  # 1 for similar, 0 for dissimilar

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModel.from_pretrained(BASE_MODEL)
    
    # Decision: Freeze some layers?
    # for param in base_model.embeddings.parameters():
    #     param.requires_grad = False

    model = EmbeddingModel(base_model)

    # Prepare datasets and dataloaders
    train_dataset = EmbeddingDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmbeddingDataset(val_texts, val_labels, tokenizer)
    
    # Decision: Batch size, sampling strategy
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Decision: Optimizer choice, learning rate
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Decision: Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * 3
    )

    # Train the model
    # Decision: Number of epochs
    train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=3, device=device)

    # Save the model
    # Decision: How to save/deploy?
    torch.save(model.state_dict(), 'fine_tuned_embedding_model.pth')

if __name__ == "__main__":
    main()