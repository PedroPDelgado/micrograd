from mlp import MLP
from loss import rmse

# train params
epochs = 200
learning_rate = 0.05

# dataset
X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# initialization
mlp = MLP((3, 4, 4, 1))

# training loop
for epoch in range(epochs):

    # forward pass
    y_pred = [mlp(x) for x in X]

    # calculate loss
    loss = rmse(ys, y_pred)
    #loss = sum((ygt - yout)**2 for ygt, yout in zip(ys, y_pred))

    # reset gradients
    for p in mlp.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()

    # update weights through gradients
    for p in mlp.parameters():
        p.data += learning_rate * -p.grad

    print(f"Epoch {epoch}: loss={loss.data}")
    print(f"\ty_pred: {[y_pred]}")