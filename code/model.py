import tensorflow as tf

def build_model():
    inputs = tf.keras.layers.Input(shape=(512, 512, 1), dtype='float64')
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)

    x = tf.keras.layers.Dropout(0.4)(conv2)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.concatenate([conv2, x], axis=-1)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)

    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.concatenate([conv1, x], axis=-1)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    output_layer = tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=output_layer)
