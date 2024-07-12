


import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time

#start time 

start = time.time()

data_dir = r"C:\Amrita\Deep Learning\DATA_LABELS_FINAL"
categories = os.listdir(data_dir)

X = []
y = []
for category in categories:
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        
        img = Image.open(img_path).convert("L")  # Convert image to grayscale
        img = img.resize((100, 100))
        
        img = np.asarray(img) / 255.0
        X.append(img)
        y.append(categories.index(category))
        
plt.imshow(X[0], cmap="gray")  # Display the first preprocessed image in grayscale

X = np.array(X)
y = np.array(y)

#start time
start = time.time()

# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data
X_train = X_train.reshape((-1, 100, 100, 1))
X_test = X_test.reshape((-1, 100, 100, 1))

# Model Architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(len(categories), activation='softmax'))

# Model Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Model Training
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

#end time 
end = time.time()
print(f"Training Time: {(end - start)/60} minutes")

# model.save("trained_model.h5")

model_json = model.to_json()
with open(r"C:\Amrita\Deep Learning\Codes\CNN\model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(r"C:\Amrita\Deep Learning\Codes\CNN\model.h5")


# Extract accuracy and loss values from history
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Print the last values of accuracy and loss during training
print("Training Accuracy:", train_accuracy[-1])
print("Training Loss:", train_loss[-1])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Testing Accuracy:", test_accuracy)
print("Testing Loss:", test_loss)

# Print the last values of accuracy and loss during validation
print("Validation Accuracy:", val_accuracy[-1])
print("Validation Loss:", val_loss[-1])


#classification report
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test, np.argmax(predictions, axis=1), target_names=categories))

#confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=categories, yticklabels=categories)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

 





# Plotting training accuracy and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting training loss and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


#end time in minutes
end = time.time()
print(f"Total Time: {(end - start)/60} minutes")



# Save the trained model as an h5 file
# model_json = model.to_json()
# with open("trained_CNN_model.json", "w") as json_file:
#     json_file.write(model_json)
    
# model.save_weights("trained_CNN_model.h5")

# sample_image_path = r"C:\Amrita\DL\ma\cropped_img68_bottom_character_62.jpg"  # Replace with the path to your sample image
# sample_image = Image.open(sample_image_path).convert("L")
# sample_image = sample_image.resize((100, 100))
# sample_image = np.asarray(sample_image) / 255.0
# sample_image = np.expand_dims(sample_image, axis=0)
# sample_image = np.expand_dims(sample_image, axis=3)

# # Make predictions on the sample image
# predictions = model.predict(sample_image)
# predicted_class_index = np.argmax(predictions, axis=1)[0]
# predicted_category = categories[predicted_class_index]

# print("Predicted category:", predicted_category)

