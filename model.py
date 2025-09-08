import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Initialize the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: acne, dry skin, oily skin
])

# Compile the model
model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training images from 'dataset/train'
train_generator = train_datagen.flow_from_directory(
    'C:/Users/gamin/Downloads/sk project/dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# Load validation images from 'dataset/validation'
validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/gamin/Downloads/sk project/dataset/valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# Train the model with validation
history = model.fit(
    train_generator,
    epochs=10,  # jitne epochs chahiye
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the trained model
model.save('skin_model.h5')
