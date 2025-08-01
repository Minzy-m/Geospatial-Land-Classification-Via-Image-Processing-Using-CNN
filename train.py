import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the directory containing your images
image_directory = r"UCMerced_LandUse\Images"

# Check if the specified directory exists
if not os.path.exists(image_directory):
    raise ValueError("The specified image directory does not exist.")

# Define the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generate the train and validation generators
train_generator = datagen.flow_from_directory(
    directory=image_directory,
    target_size=(150, 150),  # Adjust the target size based on your requirements
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=42
)

validation_generator = datagen.flow_from_directory(
    directory=image_directory,
    target_size=(150, 150),  # Adjust the target size based on your requirements
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Print class indices
print("Class indices:", train_generator.class_indices)

# Check the number of images found
print("Number of training images:", train_generator.samples)
print("Number of validation images:", validation_generator.samples)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a callback to save the model weights during training
checkpoint_callback = ModelCheckpoint('model_weights.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[checkpoint_callback])

# Save the entire model
model.save('crop.h5')
