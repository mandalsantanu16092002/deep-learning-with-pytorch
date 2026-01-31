import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)

X = torch.randn(200, 2).to(device)
y = (X[:, 0] + X[:, 1] > 0).float().view(-1, 1).to(device)

W1 = torch.randn(2, 16, requires_grad=True, device=device)
b1 = torch.zeros(16, requires_grad=True, device=device)

W2 = torch.randn(16, 1, requires_grad=True, device=device)
b2 = torch.zeros(1, requires_grad=True, device=device)

lr = 0.01
epochs = 200

for epoch in range(epochs):
   
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)

    z2 = a1 @ W2 + b2
    y_pred = torch.sigmoid(z2)

    loss = -(y * torch.log(y_pred + 1e-8) +
             (1 - y) * torch.log(1 - y_pred + 1e-8)).mean()

    loss.backward()

    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad

        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    predictions = (y_pred > 0.5).float()
    accuracy = (predictions == y).float().mean() * 100

print(f"Training Accuracy: {accuracy:.2f}%")
