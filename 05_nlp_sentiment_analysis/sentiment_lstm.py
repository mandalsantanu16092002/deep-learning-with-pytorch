import torch
import torch.nn as nn
import torch.optim as optim
from utils import build_vocab, encode_sentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
sentences = [
    "I love this movie",
    "This film is amazing",
    "What a great experience",
    "I enjoyed the story",
    "Absolutely fantastic acting",
    "I hate this movie",
    "This film is terrible",
    "Worst movie ever",
    "I did not like the plot",
    "Very boring and slow"
]

labels = [1,1,1,1,1, 0,0,0,0,0]

vocab = build_vocab(sentences)
vocab_size = len(vocab)
max_len = 6

X = torch.tensor(
    [encode_sentence(s, vocab, max_len) for s in sentences]
).to(device)

y = torch.tensor(labels).float().view(-1, 1).to(device)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n.squeeze(0))
        return self.sigmoid(out)

model = SentimentLSTM(vocab_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 300
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        preds = (outputs > 0.5).float()
        acc = (preds == y).float().mean() * 100
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

model.eval()
with torch.no_grad():
    test_sentence = "I really love this film"
    encoded = torch.tensor(
        [encode_sentence(test_sentence, vocab, max_len)]
    ).to(device)

    prediction = model(encoded)
    sentiment = "Positive" if prediction.item() > 0.5 else "Negative"

print(f"Sentence: '{test_sentence}'")
print("Predicted Sentiment:", sentiment)
