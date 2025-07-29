import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Early stopping function
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, current_loss):
        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

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

# Step 2: Define the Model (CNN-Based with Dropout)
class CNNChangepointModel(nn.Module):
    def __init__(self, sequence_length):
        super(CNNChangepointModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        # Adding a third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)  # Add batch normalization for the new layer
        self.fc1 = nn.Linear(64 * sequence_length, 128)
        self.dropout = nn.Dropout(p=0.5)  # Dropout added
        self.fc2 = nn.Linear(128, sequence_length)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1d
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))  # Pass through the new conv layer
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.dropout(self.fc1(x)))  # Dropout applied here
        x = self.fc2(x)  # Output logits for each position
        return x


# Step 3: Training Function with Early Stopping and Delta-Based Accuracy
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, patience=5, delta=5):
    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += ((predicted >= labels - delta) & (predicted <= labels + delta)).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Validate the model with delta
        val_loss, val_accuracy = evaluate_model_with_delta(model, val_loader, criterion, delta)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies


# Step 4: Evaluation Function (Includes Loss and Accuracy)
# Step 4: Evaluation Function with Delta-Based Accuracy
def evaluate_model_with_delta(model, data_loader, criterion, delta=5):
    """
    Evaluate the model and calculate accuracy, allowing for a delta tolerance
    around the true changepoint.
    """
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

            # Check if the predicted changepoint is within the delta range of the true changepoint
            correct += ((predicted >= labels - delta) & (predicted <= labels + delta)).sum().item()
    
    accuracy = 100 * correct / total
    return running_loss / len(data_loader), accuracy


# Step 5: Data Preparation (Train/Test Split)
sequence_length = 1000
num_samples = 6000
batch_size = 32

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
model = CNNChangepointModel(sequence_length)
criterion = nn.CrossEntropyLoss()  # Use cross-entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 7: Train the Model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, optimizer, criterion, num_epochs=20, patience=5, delta=5
)
# Step 8: Evaluate the Model on Test Set with Delta
test_loss, test_accuracy = evaluate_model_with_delta(model, test_loader, criterion, delta=5)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy (with delta): {test_accuracy:.2f}%")


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
