import os
import signal
import tensorflow as tf


RMSprop = tf.keras.optimizers.RMSprop
Models = tf.keras.models
Layers = tf.keras.layers
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


# defining a schema/model

model = Models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # #Filters = 16
    Layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    Layers.MaxPooling2D(2, 2),
    Layers.Conv2D(32, (3, 3), activation='relu'),
    Layers.MaxPooling2D(2, 2),
    Layers.Conv2D(64, (3, 3), activation='relu'),
    Layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    Layers.Flatten(),
    # 512 neuron hidden layer,
    Layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    Layers.Dense(1, activation='sigmoid')
])

# to see the shapes and parameters.
model.summary()

# configuring the specs for model training
model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# defining the directories
# Note:
# The folder structure is
# Data (name anything)
#   - train
#       - cats
#       - dogs
#   -validation
#       - cats
#       - dogs

base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Data preprocessing

# creating the object of class ImageDataGenerator
# also providing a scaling factor 1/255 (normalising )
# image pixel has 8 bits -> range [0,255]
# Hence normalising the pixels to the range [0,1]
train_datagen = ImageDataGenerator(rescale=1.0/255)
validate_datagen = ImageDataGenerator(rescale=1.0/255)

# creating a generators by pointing to the required directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    class_mode='binary',  # binary classification
    target_size=(150, 150)  # resizing on the fly
)

validate_generator = validate_datagen.flow_from_directory(
    validation_dir,
    batch_size=20,
    class_mode='binary',  # binary classification
    target_size=(150, 150)  # resizing on the fly
)


# training : by using model.fit()
# returns the history

history = model.fit(
    train_generator,
    validation_data=validate_generator,
    steps_per_epoch=100,
    validation_steps=50,
    verbose=2
)

# To free up the resources.
#TODO: read about it
os.kill(os.getpid(),
        signal.SIGTERM
        )

#TODO: Save the model do that we can train once and predict many times.