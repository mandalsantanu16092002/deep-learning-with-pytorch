import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)

X = torch.randn(500, 2)
y = (X[:, 0] + X[:, 1] > 0).float().view(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

        train_pred = (outputs > 0.5).float()
        val_pred = (val_outputs > 0.5).float()

        train_acc = (train_pred == y_train).float().mean() * 100
        val_acc = (val_pred == y_val).float().mean() * 100

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f} | "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%"
        )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pt")

os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")

loaded_model = MLP().to(device)
loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
loaded_model.eval()

with torch.no_grad():
    sample = torch.tensor([[0.5, -1.2]], device=device)
    prediction = loaded_model(sample)

print("Sample prediction:", prediction.item())
