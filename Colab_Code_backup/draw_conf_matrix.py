# Geenration of confusion matrix

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Switch to evaluation mode
model.eval()
all_preds = []
all_labels = []

# Final inference on the Testing Set (the 5%)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Final Accuracy
cm = confusion_matrix(all_labels, all_preds)
accuracy = (cm[0,0] + cm[1,1]) / sum(sum(cm))

# Visualization of the Confusion Matrix (similar to Figure 7 in paper)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Clean (NDRC)', 'Violation (DRCV)'],
            yticklabels=['Clean (NDRC)', 'Violation (DRCV)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Final Testing Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')
plt.show()

# Print full report (Recall and Precision)
print("\nFinal Performance Report:")
print(classification_report(all_labels, all_preds, target_names=['Clean', 'Violation']))