import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Simple model for testing
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create synthetic dataset
    num_samples = 1000
    input_dim = 10
    x = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, 10)  # Random targets
    
    # Create DataLoader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = TestModel()
    
    # Compile the model
    model = torch.compile(
        model,
        mode="default",
        dynamic=True,
        fullgraph=True
    )
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 