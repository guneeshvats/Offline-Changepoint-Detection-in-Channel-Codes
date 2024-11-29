import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Data
def generate_bernoulli_sequence(length, p1, p2, changepoint):
    """
    Generates a binary sequence with two different Bernoulli distributions.
    """
    first_part = np.random.binomial(1, p1, size=changepoint)
    second_part = np.random.binomial(1, p2, size=length - changepoint)
    sequence = np.concatenate([first_part, second_part])
    return sequence, changepoint

def generate_synthetic_data(num_samples, sequence_length):
    """
    Generates multiple samples of binary sequences with random changepoints.
    """
    sequences = []
    labels = []
    for _ in range(num_samples):
        changepoint = np.random.randint(low=int(sequence_length * 0.3), high=int(sequence_length * 0.7))
        sequence, label = generate_bernoulli_sequence(sequence_length, p1=0.2, p2=0.8, changepoint=changepoint)
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Step 2: Define Transformer-Based Model for Changepoint Detection
class TransformerChangepointModel(nn.Module):
    def __init__(self, sequence_length, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.3):
        super(TransformerChangepointModel, self).__init__()
        self.embedding = nn.Linear(1, d_model)  # Embedding from input to model dimension
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model * sequence_length, sequence_length)  # Output is sequence length
        self.layer_norm = nn.LayerNorm(d_model)  # Layer Normalization
        
    def forward(self, x):
        x = x.unsqueeze(-1)  # Add feature dimension for embedding
        x = self.embedding(x)
        
        # Transformer expects input as (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.layer_norm(x)  # Apply layer normalization
        x = self.transformer_encoder(x)  # Transformer encoder
        x = x.permute(1, 0, 2).contiguous()  # Back to (batch_size, seq_len, d_model)
        
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # Apply dropout before the final layer
        x = self.fc(x)
        return x


# Step 3: Training Function
# Training function with learning rate scheduler
# Training function with gradient clipping
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=1):
    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # Update weights
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Validate the model
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Step 4: Evaluation Function
def evaluate_model(model, data_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(data_loader), accuracy

# Step 5: Data Preparation (Train/Test Split)
sequence_length = 100
num_samples = 10000
batch_size = 16

# Generate synthetic data
sequences, labels = generate_synthetic_data(num_samples, sequence_length)

# Convert to PyTorch tensors
sequences = torch.tensor(sequences, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Create Dataset
dataset = TensorDataset(sequences, labels)

# Split the data into training, validation, and test sets (70% Train, 15% Val, 15% Test)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 6: Initialize Model, Loss, and Optimizer
model = TransformerChangepointModel(sequence_length)
criterion = nn.CrossEntropyLoss()  # Use cross-entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


# Step 7: Train the Model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=20
)

# Step 8: Evaluate the Model on Test Set
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Step 9: Plot Training/Validation Loss and Accuracy
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Step 10: Example Prediction
def post_process(predictions):
    """
    Post-process the output to ensure only one changepoint is predicted.
    """
    max_index = torch.argmax(predictions, dim=1)  # Get the index with the highest score
    return max_index

# Generate a new test sequence
test_sequence, true_changepoint = generate_bernoulli_sequence(sequence_length, p1=0.2, p2=0.8, changepoint=60)
test_sequence = torch.tensor(test_sequence, dtype=torch.float32).unsqueeze(0)

# Model prediction
model.eval()
with torch.no_grad():
    output = model(test_sequence)
    changepoint_prediction = post_process(output)

# Print Results
print(f"True Changepoint: {true_changepoint}")
print(f"Predicted Changepoint: {changepoint_prediction.item()}")

# Plot the test sequence
plt.plot(test_sequence.squeeze().numpy())
plt.axvline(x=true_changepoint, color='r', linestyle='--', label="True Changepoint")
plt.axvline(x=changepoint_prediction.item(), color='g', linestyle='-', label="Predicted Changepoint")
plt.legend()
plt.show()
