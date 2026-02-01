
# Problem  -> Each sample is a sequence of numbers. If sum(sequence) > 0 → class 1, else class 0. This keeps focus on sequence learning, not data loading.

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)

num_samples = 500
seq_length = 10
input_size = 1

X = torch.randn(num_samples, seq_length, input_size)
y = (X.sum(dim=1) > 0).float()

X, y = X.to(device), y.to(device)

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return self.sigmoid(out)

model = RNNModel().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        preds = (outputs > 0.5).float()
        acc = (preds == y).float().mean() * 100
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")
