import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

# Use pre-trained BERT model
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

def train():
    # Tokenize the data (for AG News dataset)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenize_data = lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)

    # Load the AG News dataset
    train_dataset = load_dataset('ag_news')['train'].map(tokenize_data, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Define loss function and optimizer
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss() # redundant as BERT model already comes with a loss function

    print('Training BERT model...')
    # Training loop
    for epoch in range(3):
        bert_model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels =  batch['label'].to(device)

            # Forward pass
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/3 completed with loss: {loss.item():.4f}')

    # Save the trained model
    bert_model.save_pretrained('./outputs/bert_model')
    tokenizer.save_pretrained('./outputs/bert_model')

def test():
    model = BertForSequenceClassification.from_pretrained('./outputs/bert_model')
    tokenizer = BertTokenizer.from_pretrained('./outputs/bert_model')
    tokenize_data = lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)

    test_dataset = load_dataset('ag_news')['test'].map(tokenize_data, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels =  batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy: {accuracy:.4f}')