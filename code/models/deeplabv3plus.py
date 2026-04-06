import tensorflow as tf

def build_deeplabv3plus(input_shape=(512, 512, 1), num_classes=4):
    """Build DeepLabV3+ architecture for image segmentation.
    
    DeepLabV3+ is a state-of-the-art semantic segmentation model that uses
    atrous convolution and spatial pyramid pooling.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of segmentation classes
    
    Returns:
        tf.keras.Model: DeepLabV3+ model
    """
    inputs = tf.keras.layers.Input(shape=input_shape, dtype='float64')
    
    # Encoder (Modified ResNet backbone)
    # Entry flow
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Save low-level features for decoder
    low_level_features = x
    
    # Strided convolution blocks
    x = _seperable_conv_block(x, 128, stride=2)
    x = _seperable_conv_block(x, 128, stride=1)
    x = _seperable_conv_block(x, 128, stride=1)
    
    x = _seperable_conv_block(x, 256, stride=2)
    x = _seperable_conv_block(x, 256, stride=1)
    x = _seperable_conv_block(x, 256, stride=1)
    
    x = _seperable_conv_block(x, 512, stride=2)
    x = _seperable_conv_block(x, 512, stride=1)
    x = _seperable_conv_block(x, 512, stride=1)
    
    # Atrous Spatial Pyramid Pooling (ASPP)
    aspp_out = _aspp(x)
    
    # Decoder
    # Upsample ASPP output
    decoder_x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(aspp_out)
    
    # Process low-level features
    low_level_features = tf.keras.layers.Conv2D(48, (1, 1), padding='same', use_bias=False)(low_level_features)
    low_level_features = tf.keras.layers.BatchNormalization()(low_level_features)
    low_level_features = tf.keras.layers.Activation('relu')(low_level_features)
    
    # Concatenate
    decoder_x = tf.keras.layers.Concatenate()([decoder_x, low_level_features])
    
    # Refine features
    decoder_x = _seperable_conv_block(decoder_x, 256, stride=1)
    decoder_x = _seperable_conv_block(decoder_x, 256, stride=1)
    
    # Final upsampling and classification
    decoder_x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(decoder_x)
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(decoder_x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='DeepLabV3Plus')


def _seperable_conv_block(x, filters, stride=1):
    """Depthwise separable convolution block."""
    x = tf.keras.layers.SeparableConv2D(filters, (3, 3), strides=(stride, stride), 
                                        padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def _aspp(x):
    """Atrous Spatial Pyramid Pooling module."""
    # Image pooling branch
    pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    pool = tf.keras.layers.Reshape((1, 1, -1))(pool)
    pool = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(pool)
    pool = tf.keras.layers.BatchNormalization()(pool)
    pool = tf.keras.layers.Activation('relu')(pool)
    pool = tf.keras.layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(pool)
    
    # 1x1 convolution branch
    conv1x1 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    conv1x1 = tf.keras.layers.BatchNormalization()(conv1x1)
    conv1x1 = tf.keras.layers.Activation('relu')(conv1x1)
    
    # 3x3 convolutions with different dilation rates
    conv3x3_1 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', dilation_rate=(6, 6), use_bias=False)(x)
    conv3x3_1 = tf.keras.layers.BatchNormalization()(conv3x3_1)
    conv3x3_1 = tf.keras.layers.Activation('relu')(conv3x3_1)
    
    conv3x3_2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', dilation_rate=(12, 12), use_bias=False)(x)
    conv3x3_2 = tf.keras.layers.BatchNormalization()(conv3x3_2)
    conv3x3_2 = tf.keras.layers.Activation('relu')(conv3x3_2)
    
    conv3x3_3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', dilation_rate=(18, 18), use_bias=False)(x)
    conv3x3_3 = tf.keras.layers.BatchNormalization()(conv3x3_3)
    conv3x3_3 = tf.keras.layers.Activation('relu')(conv3x3_3)
    
    # Concatenate all branches
    concat = tf.keras.layers.Concatenate()([pool, conv1x1, conv3x3_1, conv3x3_2, conv3x3_3])
    
    # Project to 256 channels
    output = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(concat)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Activation('relu')(output)
    
    return output