import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

path = "Dataset/my_real_vs_ai_dataset/my_real_vs_ai_dataset"
trainData = tf.keras.utils.image_dataset_from_directory(path,shuffle=True,image_size=(128, 128),batch_size=32,validation_split=0.2,subset="training")
testData = tf.keras.utils.image_dataset_from_directory(path,shuffle=False,image_size=(128, 128),batch_size=32,validation_split=0.2,subset="validation")

trainData = trainData.map(lambda x,y: (x/255.0,y), num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=500).prefetch(tf.data.AUTOTUNE)
testData = testData.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    # Feature extraction
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    # Classification , 0=ai & 1=human
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("bestModel.keras",monitor="val_accuracy",save_best_only=True,verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3,restore_best_weights=True,verbose=1)
]

history = model.fit(trainData,validation_data=testData,epochs=10,callbacks=callbacks)
loss, accuracy = model.evaluate(testData)
model.save("finalModel.keras")

print(f"Final val accuracy: {accuracy:.4f}")
print(f"Final val loss: {loss:.4f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],label='Train')
ax1.plot(history.history['testing accuracy'],label='Val')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'],label='Train')
ax2.plot(history.history['testing loss'],label='Val')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()