import tensorflow as tf
import numpy as np
from PIL import Image
import os


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# 📦 Load dataset (only needed for class names)
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=8
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=8
)
class_names = dataset.class_names
print("Classes:", class_names)

# ================================
# 🧠 LOAD OR TRAIN MODEL
# ================================

if os.path.exists("plant_model.h5"):
    print("📂 Loading existing model...")
    model = tf.keras.models.load_model("plant_model.h5")

else:
    print("🧠 Training new model...")

    model = tf.keras.Sequential([data_augmentation,   # 🔥 NEW

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(dataset, validation_data=val_dataset, epochs=15)
    model.save("plant_model.h5")

    print("✅ Model trained and saved!")

# ================================
# 🔍 TEST IMAGE
# ================================

img = Image.open("test.jpg").convert("RGB")
img = img.resize((224, 224))

img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

# 🎯 Predict
predictions = model.predict(img_array)
score = np.argmax(predictions)

print("\n🌿 Prediction:", class_names[score])
print("📊 Confidence:", np.max(predictions))


# ================================
# 🧪 ALL PROBABILITIES
# ================================

print("\n🧪 All probabilities:")
for i, plant in enumerate(class_names):
    print(f"{plant}: {predictions[0][i]:.2f}")


# ================================
# 🏆 TOP 2
# ================================

sorted_indices = np.argsort(predictions[0])[::-1]

top1 = class_names[sorted_indices[0]]
top2 = class_names[sorted_indices[1]]

print("\n🌿 Top match:", top1)
print("🌱 Second guess:", top2)


# ================================
# 🎯 VERIFICATION MODE
# ================================

target_plant = "curry"

target_index = class_names.index(target_plant)
target_confidence = predictions[0][target_index]

print(f"\n🔍 Match confidence for {target_plant}: {target_confidence:.2f}")

if target_confidence > 0.6:
    print("✅ Strong match!")
elif target_confidence > 0.3:
    print("⚠️ Possible match, but not very confident")
else:
    print("❌ Likely NOT the plant")