# Model Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NCSU_DRCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Loading and Splitting
dataset = DRCDataset(root_dir='data', transform=data_transforms)
train_size = int(0.80 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Total samples: {len(dataset)}")
print(f"Training: {len(train_data)} | Validation: {len(val_data)} | Testing: {len(test_data)}")


def train(epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Simple Accuracy Check
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, pred = torch.max(model(imgs), 1)
                correct += (pred == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {100*correct/len(val_data):.2f}%")

train(epochs=20)

# Saving model
torch.save(model.state_dict(), 'ncsu_drcnn_weights.pth')
print("Training complete! Model saved to 'ncsu_drcnn_weights.pth'")