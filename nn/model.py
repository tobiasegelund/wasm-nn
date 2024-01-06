import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
        # return x


data = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (4, 5),
    (5, 0),
    (0, 1),
    (0, 1),
    (5, 0),
    (0, 0),
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]
data = torch.tensor(data)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

NUM_EPOCHS = 10
X = data[:, 0].reshape(-1, 1).float()
y = data[:, 1].long()

for epoch in range(NUM_EPOCHS):
    outputs = model(X)

    # Compute the loss
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item()}")

model.eval()

print(model.forward(x=X[:1, 0].reshape(-1, 1)))

torch.onnx.export(
    model,
    X[:1, :],
    "nn.onnx",
    export_params=True,
    opset_version=10,
    input_names=["input"],
    output_names=["output"],
)
