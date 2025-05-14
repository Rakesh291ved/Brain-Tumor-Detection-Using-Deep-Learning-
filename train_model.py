import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Paths
data_dir = './brain_tumor_dataset'
model_path = './models/bt_resnet50_model.pt'

# Transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = True

n_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(n_inputs, 2048),
    nn.SELU(),
    nn.Dropout(0.4),
    nn.Linear(2048, 2048),
    nn.SELU(),
    nn.Dropout(0.4),
    nn.Linear(2048, 4),
    nn.LogSigmoid()
)

model.to(device)

# Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop (simple, 5 epochs)
epochs = 5
model.train()
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Check if weights are updated
        for param in model.parameters():
            if param.grad is not None:
                print(f"Parameter {param.size()} has been updated.")  # Debug log

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save model
try:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("✅ Model saved successfully to", model_path)
except Exception as e:
    print(f"Error saving model: {e}")
