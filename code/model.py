import tensorflow as tf


def encoder_block(inputs, filters, dropout_rate=0.0, block_name="encoder"):
    """
    Encoder block for U-Net: performs downsampling with convolutions.
    
    Args:
        inputs: Input tensor
        filters: Number of filters for convolution layers
        dropout_rate: Dropout rate (0.0 means no dropout)
        block_name: Name prefix for the layers
    
    Returns:
        conv_output: Output of convolutions (used for skip connections)
        pooled_output: Output after max pooling (fed to next encoder block)
    """
    # First convolution layer
    conv = tf.keras.layers.Conv2D(
        filters=filters, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu',
        name=f"{block_name}_conv1"
    )(inputs)
    
    # Batch normalization for better training stability
    conv = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn1")(conv)
    
    # Second convolution layer
    conv = tf.keras.layers.Conv2D(
        filters=filters, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu',
        name=f"{block_name}_conv2"
    )(conv)
    
    # Batch normalization
    conv_output = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn2")(conv)
    
    # Apply dropout if specified
    if dropout_rate > 0:
        conv_output = tf.keras.layers.Dropout(dropout_rate, name=f"{block_name}_dropout")(conv_output)
    
    # Max pooling for downsampling (2x2 reduces spatial dimensions by half)
    pooled_output = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name=f"{block_name}_pool"
    )(conv_output)
    
    return conv_output, pooled_output


def bottleneck(inputs, filters, dropout_rate=0.4):
    """
    Bottleneck layer: the deepest part of the U-Net.
    
    Args:
        inputs: Input tensor from the last encoder block
        filters: Number of filters for convolution layers
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Output tensor after convolutions and dropout
    """
    # First convolution in bottleneck
    x = tf.keras.layers.Conv2D(
        filters=filters, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu',
        name="bottleneck_conv1"
    )(inputs)
    
    # Batch normalization
    x = tf.keras.layers.BatchNormalization(name="bottleneck_bn1")(x)
    
    # Second convolution in bottleneck
    x = tf.keras.layers.Conv2D(
        filters=filters, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu',
        name="bottleneck_conv2"
    )(x)
    
    # Batch normalization
    x = tf.keras.layers.BatchNormalization(name="bottleneck_bn2")(x)
    
    # Dropout for regularization
    x = tf.keras.layers.Dropout(dropout_rate, name="bottleneck_dropout")(x)
    
    return x


def decoder_block(inputs, skip_connection, filters, block_name="decoder"):
    """
    Decoder block for U-Net: performs upsampling and concatenation with skip connections.
    
    Args:
        inputs: Input tensor from previous decoder block or bottleneck
        skip_connection: Tensor from corresponding encoder block (skip connection)
        filters: Number of filters for convolution layers
        block_name: Name prefix for the layers
    
    Returns:
        Output tensor after upsampling, concatenation, and convolutions
    """
    # Transpose convolution for upsampling (2x2 doubles spatial dimensions)
    upsampled = tf.keras.layers.Conv2DTranspose(
        filters=filters, 
        kernel_size=(3, 3), 
        padding='same', 
        strides=(2, 2), 
        activation='relu',
        name=f"{block_name}_transpose"
    )(inputs)
    
    # Concatenate with skip connection from encoder
    # This helps preserve fine-grained details from earlier layers
    concatenated = tf.keras.layers.concatenate(
        [skip_connection, upsampled], 
        axis=-1,
        name=f"{block_name}_concat"
    )
    
    # First convolution after concatenation
    conv = tf.keras.layers.Conv2D(
        filters=filters, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu',
        name=f"{block_name}_conv1"
    )(concatenated)
    
    # Batch normalization
    conv = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn1")(conv)
    
    # Second convolution
    conv = tf.keras.layers.Conv2D(
        filters=filters, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu',
        name=f"{block_name}_conv2"
    )(conv)
    
    # Batch normalization
    conv = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn2")(conv)
    
    return conv


def build_model():
    """
    Build a U-Net model for COVID-19 lung CT scan segmentation.
    
    This model uses a U-Net architecture with:
    - Encoder path: 3 levels of downsampling (64 -> 128 -> 256 filters)
    - Bottleneck: 256 filters with dropout
    - Decoder path: 3 levels of upsampling with skip connections
    - Output: 4-class segmentation (softmax activation)
    
    Input shape: (512, 512, 1) - grayscale CT scan slices
    Output shape: (512, 512, 4) - probability maps for 4 segmentation classes
    
    Returns:
        tf.keras.Model: Compiled U-Net model
    """
    # Input layer: single channel (grayscale) 512x512 CT scan images
    inputs = tf.keras.layers.Input(shape=(512, 512, 1), dtype='float64', name="input_ct_scan")
    
    # ============ ENCODER PATH ============
    # Encoder block 1: 512x512x1 -> 512x512x64 (skip) and 256x256x64 (pooled)
    encoder1_skip, encoder1_pooled = encoder_block(
        inputs, 
        filters=64, 
        dropout_rate=0.0,  # No dropout in first block
        block_name="encoder1"
    )
    
    # Encoder block 2: 256x256x64 -> 256x256x128 (skip) and 128x128x128 (pooled)
    encoder2_skip, encoder2_pooled = encoder_block(
        encoder1_pooled, 
        filters=128, 
        dropout_rate=0.4,  # Add dropout for regularization
        block_name="encoder2"
    )
    
    # ============ BOTTLENECK ============
    # Bottleneck: 128x128x128 -> 128x128x256
    # This is the deepest part of the network
    bottleneck_output = bottleneck(encoder2_pooled, filters=256, dropout_rate=0.4)
    
    # ============ DECODER PATH ============
    # Decoder block 1: 128x128x256 -> 256x256x128
    # Concatenates with encoder2_skip for a total of 256x256x(128+128)
    decoder1 = decoder_block(
        bottleneck_output, 
        encoder2_skip, 
        filters=128,
        block_name="decoder1"
    )
    
    # Decoder block 2: 256x256x128 -> 512x512x64
    # Concatenates with encoder1_skip for a total of 512x512x(64+64)
    decoder2 = decoder_block(
        decoder1, 
        encoder1_skip, 
        filters=64,
        block_name="decoder2"
    )
    
    # ============ OUTPUT LAYER ============
    # Final convolution to produce 4-class segmentation map
    # Uses softmax activation for multi-class classification
    output_layer = tf.keras.layers.Conv2D(
        filters=4,  # 4 classes for segmentation
        kernel_size=(3, 3), 
        padding='same', 
        activation='softmax',
        name="output_segmentation"
    )(decoder2)
    
    # Create and return the model
    model = tf.keras.Model(inputs=inputs, outputs=output_layer, name="covid_lung_segmentation_unet")
    
    return model
