import numpy as np
import matplotlib.pyplot as plt

models = ['CNN', 'AlexNet', 'LeNet5', 'RNN', 'LSTM']
train_accuracies = [93, 6, 96, 72, 88]
test_accuracies = [92, 5.4, 88, 63, 82]
val_accuracies = [90, 4.6, 85, 62, 82]

bar_width = 0.25

r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

color_train = '#e0ac69'
color_test = '#d1b5ff'
color_val = '#e59358'

plt.bar(r1, train_accuracies, width=bar_width, label='Train Accuracies', color=color_train)
plt.bar(r2, test_accuracies, width=bar_width, label='Test Accuracies', color=color_test)
plt.bar(r3, val_accuracies, width=bar_width, label='Validation Accuracies', color=color_val)

plt.title('Training, Test, and Validation Accuracies')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')

plt.xticks([r + bar_width for r in range(len(models))], models)

for i, value in enumerate(train_accuracies):
    plt.text(i, value + 1, str(value) + '%', ha='center', va='bottom', color='black')
for i, value in enumerate(test_accuracies):
    plt.text(i + bar_width, value + 1, str(value) + '%', ha='center', va='bottom', color='black')
for i, value in enumerate(val_accuracies):
    plt.text(i + 2 * bar_width, value + 1, str(value) + '%', ha='center', va='bottom', color='black')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3)

plt.show()
