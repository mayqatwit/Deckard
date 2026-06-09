import sys
import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt # type: ignore

def buildLayers(kernel: tuple) -> list:
    """Returns a list of layer creation functions to dynamically add them to the model later"""
    return [
        lambda: layers.MaxPooling2D(),
        lambda: layers.Conv2D(32, kernel, activation='relu', input_shape=(128, 128, 3)),
        lambda: layers.Conv2D(64, kernel, activation='relu'),
        lambda: layers.Conv2D(128, kernel, activation='relu'),
        lambda: layers.Flatten(),
        lambda: layers.Dense(128, activation='relu'),
        lambda: layers.Dropout(0.5),
        lambda: layers.Dense(1, activation='sigmoid'),
        lambda: tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet'),
        lambda: layers.GlobalAveragePooling2D(),
        lambda: layers.Dense(256, activation='relu')
    ]

def importData(path: str):
    """Seperates tesing and training data based on a given directory path. The directory must have 2 orther directories in it, one real and one ai."""
    trainData = tf.keras.utils.image_dataset_from_directory(path,shuffle=True,image_size=(128, 128),batch_size=32,validation_split=0.2,subset="training", seed=69)
    testData = tf.keras.utils.image_dataset_from_directory(path,shuffle=False,image_size=(128, 128),batch_size=32,validation_split=0.2,subset="validation", seed=69)
    trainData = trainData.map(lambda x,y: (x/255.0,y), num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=500).prefetch(tf.data.AUTOTUNE)
    testData = testData.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return trainData, testData

def chooseLayers(layers: list,args: list):
    """Given a pre-made list of layers creation functions and a list of arguments, returns a list of chosen layers from the pre-made list"""
    layerChoices = []
    for i in range(0,len(args)):
        layerChoices.append(layers[int(args[i])]())
    return layerChoices

def trainModel(layers: list, numEpochs: int, trainData, testData, modelName: str):
    """Trains and tests a model based on a given list of layers, number of epochs, and the training and testing data"""
    for layer in layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = False
    model = tf.keras.Sequential(layers)
    model.summary()
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("bestModel.keras",monitor="val_accuracy",save_best_only=True,verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=3,restore_best_weights=True,verbose=1)
    ]

    history = model.fit(trainData,validation_data=testData,epochs=numEpochs,callbacks=callbacks)
    loss, accuracy = model.evaluate(testData)
    model.save(modelName)

    return history, loss, accuracy

def  plotResults(history, plotName: str):
    """Plots the result of training the model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],label='Train')
    ax1.plot(history.history['val_accuracy'],label='Val')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'],label='Train')
    ax2.plot(history.history['val_loss'],label='Val')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plotName)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) >= 7:
        path = sys.argv[1]
        modelName = sys.argv[2]
        plotName = sys.argv[3]
        epochs = int(sys.argv[4])
        kernel = (int(sys.argv[5]),int(sys.argv[5]))

        trainData, testData = importData(path)
        chosen = chooseLayers(buildLayers(kernel), sys.argv[6:])
        
        history, loss, accuracy = trainModel(chosen, epochs, trainData, testData, modelName)

        print(f"Final val accuracy: {accuracy:.4f}")
        print(f"Final val loss: {loss:.4f}")

        plotResults(history, plotName)

    else:
        print("\nUsage: modelTraining.py <path> <final model name> <graph image name> <epochs> <kernel size> <layer choices (multiple ints)>\n")
        print("Layer Choices:  \n\
        0: MaxPooling2D(), \n\
        1: Conv2D(32, kernel, activation='relu', input_shape=(128, 128, 3)) \n\
        2: Conv2D(64, kernel, activation='relu') \n\
        3: Conv2D(128, kernel, activation='relu') \n\
        4: Flatten() \n\
        5: Dense(128, activation='relu') \n\
        6: Dropout(0.5) \n\
        7: Dense(1, activation='sigmoid')\n\
        8: MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights=\'imagenet\')\n\
        9: GlobalAveragePooling2D()\n\
        10: Dense(256, activation='relu')")
        print("Example usage: modelTraining.py Dataset/my_real_vs_ai_dataset/my_real_vs_ai_dataset Models/finalModel.keras plot.png 2 3 1 0 0 0 2 0 3 0 4 5 6 7 \n\
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n\
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃\n\
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n\
│ conv2d (Conv2D)                      │ (None, 126, 126, 32)        │             896 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ max_pooling2d (MaxPooling2D)         │ (None, 63, 63, 32)          │               0 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 31, 31, 32)          │               0 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 15, 15, 32)          │               0 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ conv2d_1 (Conv2D)                    │ (None, 13, 13, 64)          │          18,496 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 6, 6, 64)            │               0 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ conv2d_2 (Conv2D)                    │ (None, 4, 4, 128)           │          73,856 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ max_pooling2d_4 (MaxPooling2D)       │ (None, 2, 2, 128)           │               0 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ flatten (Flatten)                    │ (None, 512)                 │               0 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ dense (Dense)                        │ (None, 128)                 │          65,664 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ dropout (Dropout)                    │ (None, 128)                 │               0 │\n\
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n\
│ dense_1 (Dense)                      │ (None, 1)                   │             129 │\n\
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n\
Total params: 159,041 (621.25 KB) \n\
Trainable params: 159,041 (621.25 KB)\n\
Non-trainable params: 0 (0.00 B)\n\
Epoch 1/2")