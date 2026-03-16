import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from imbalanced_losses import LossWarmupWrapper, SigmoidFocalLoss, SmoothAPLoss

torch.manual_seed(0)

# 5% positive rate — a realistic fraud-detection setup
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    weights=[0.95, 0.05],
    flip_y=0.02,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # [N, 1]
y_val_np = y_val  # keep as numpy for sklearn metrics

print(f"Train size: {len(X_train)}, positives: {int(y_train.sum())}")

EPOCHS = 25


def make_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
    return model, torch.optim.Adam(model.parameters(), lr=1e-3)


def eval_aucpr(model):
    model.eval()
    with torch.no_grad():
        return average_precision_score(y_val_np, model(X_val).squeeze().numpy())


# BCE baseline
model, optimizer = make_model()
loss_fn = nn.BCEWithLogitsLoss()

for _ in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss_fn(model(X_train), y_train).backward()
    optimizer.step()

print(f"BCE  AUCPR: {eval_aucpr(model):.4f}")


# Focal Loss
model, optimizer = make_model()
loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

for _ in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss_fn(model(X_train), y_train).backward()
    optimizer.step()

print(f"Focal AUCPR: {eval_aucpr(model):.4f}")


# Smooth-AP with warmup
model, optimizer = make_model()
loss_fn = LossWarmupWrapper(
    warmup_loss=nn.BCEWithLogitsLoss(),
    main_loss=SmoothAPLoss(num_classes=1, queue_size=512),
    warmup_epochs=5,
    blend_epochs=2,
    temp_start=0.1,
    temp_end=0.01,
    temp_decay_steps=1000,
)

for epoch in range(EPOCHS):
    loss_fn.on_train_epoch_start(epoch)
    model.train()
    loss_fn.on_train_batch_start(epoch)
    optimizer.zero_grad()
    loss_fn(model(X_train), y_train).backward()
    optimizer.step()

    aucpr = eval_aucpr(model)
    phase = "warmup" if loss_fn.in_warmup else ("blend" if loss_fn.in_blend else "AP")
    t = loss_fn.current_temperature
    temp_str = f"  temp={t:.4f}" if (t is not None and not loss_fn.in_warmup) else ""
    print(f"Epoch {epoch:2d} [{phase:6s}]  AUCPR={aucpr:.4f}{temp_str}")
