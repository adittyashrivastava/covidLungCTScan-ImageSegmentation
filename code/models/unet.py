import tensorflow as tf

def build_unet(input_shape=(512, 512, 1), num_classes=4):
    """Build U-Net architecture for image segmentation.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of segmentation classes
    
    Returns:
        tf.keras.Model: U-Net model
    """
    inputs = tf.keras.layers.Input(shape=input_shape, dtype='float64')
    
    # Encoder path
    # Block 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Block 2
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Block 3 (Bottleneck)
    x = tf.keras.layers.Dropout(0.4)(conv2)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Decoder path
    # Block 4
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.concatenate([conv2, x], axis=-1)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Block 5
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.concatenate([conv1, x], axis=-1)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Output layer
    output_layer = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(3, 3), padding='same', activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=output_layer, name='UNet')
