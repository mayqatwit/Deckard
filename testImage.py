import tensorflow as tf

model = tf.keras.models.load_model("bestModel.keras")
imagePath = "frame_009.jpg"

img = tf.keras.utils.load_img(imagePath, target_size=(128, 128))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, axis=0)
prediction = model.predict(img_array, verbose=0)

if (prediction[0][0] < 0.4):
    print(f"Predicted: AI ({prediction[0][0]:.4f})")
elif (prediction[0][0] > 0.6):
    print(f"Predicted: Real ({prediction[0][0]:.4f})")
else: 
    print(f"Predicted: Uncertain ({prediction[0][0]:.4f})")