import tensorflow as tf


def encoder_block(inputs, num_filters, block_name, dropout_rate=0.0):
    """
    Encoder block with two convolutional layers and optional dropout.
    
    Args:
        inputs: Input tensor
        num_filters: Number of filters for convolution layers
        block_name: Name prefix for the block (e.g., 'encoder_block_1')
        dropout_rate: Dropout rate (0.0 means no dropout)
    
    Returns:
        conv_output: Output of the encoder block (for skip connections)
        pool_output: Downsampled output (for next encoder block)
    """
    # First convolution layer in the block
    conv1 = tf.keras.layers.Conv2D(
        filters=num_filters, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu',
        name=f'{block_name}_conv1'
    )(inputs)
    
    # Second convolution layer in the block
    conv2 = tf.keras.layers.Conv2D(
        filters=num_filters, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu',
        name=f'{block_name}_conv2'
    )(conv1)
    
    # Apply dropout if specified
    if dropout_rate > 0:
        conv2 = tf.keras.layers.Dropout(dropout_rate, name=f'{block_name}_dropout')(conv2)
    
    # Max pooling for downsampling (except for the last encoder block)
    pool = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), 
        name=f'{block_name}_pool'
    )(conv2)
    
    return conv2, pool


def decoder_block(inputs, skip_connection, num_filters, block_name):
    """
    Decoder block with transpose convolution, skip connection, and two conv layers.
    
    Args:
        inputs: Input tensor from previous decoder block or bottleneck
        skip_connection: Tensor from corresponding encoder block
        num_filters: Number of filters for convolution layers
        block_name: Name prefix for the block (e.g., 'decoder_block_1')
    
    Returns:
        Output tensor of the decoder block
    """
    # Upsampling with transpose convolution
    upconv = tf.keras.layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=(3, 3),
        padding='same',
        strides=(2, 2),
        activation='relu',
        name=f'{block_name}_upconv'
    )(inputs)
    
    # Concatenate with skip connection from encoder
    concat = tf.keras.layers.concatenate(
        [skip_connection, upconv], 
        axis=-1,
        name=f'{block_name}_concat'
    )
    
    # First convolution layer after concatenation
    conv1 = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name=f'{block_name}_conv1'
    )(concat)
    
    # Second convolution layer
    conv2 = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name=f'{block_name}_conv2'
    )(conv1)
    
    return conv2


def bottleneck_block(inputs, num_filters):
    """
    Bottleneck block at the bottom of the U-Net.
    
    Args:
        inputs: Input tensor from last encoder block
        num_filters: Number of filters for convolution layers
    
    Returns:
        Output tensor of the bottleneck block
    """
    # First convolution in bottleneck
    conv1 = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='bottleneck_conv1'
    )(inputs)
    
    # Second convolution in bottleneck
    conv2 = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='bottleneck_conv2'
    )(conv1)
    
    # Dropout for regularization in bottleneck
    dropout = tf.keras.layers.Dropout(0.4, name='bottleneck_dropout')(conv2)
    
    return dropout


def build_model():
    """
    Build U-Net model for COVID lung CT scan segmentation.
    
    The model follows an encoder-decoder architecture with skip connections:
    - Encoder: Progressively downsamples while increasing feature channels (64->128->256)
    - Bottleneck: Processes the most compressed representation
    - Decoder: Progressively upsamples while decreasing feature channels (256->128->64)
    - Skip connections: Preserve spatial information from encoder stages
    
    Returns:
        tf.keras.Model: Compiled U-Net model
    """
    # Input layer - single channel CT scan image
    inputs = tf.keras.layers.Input(
        shape=(512, 512, 1), 
        dtype='float64',
        name='input_ct_scan'
    )
    
    # ===== ENCODER PATH =====
    # Encoder Block 1: 512x512x1 -> 512x512x64 -> 256x256x64
    encoder_1_skip, encoder_1_pool = encoder_block(
        inputs, 
        num_filters=64, 
        block_name='encoder_block_1'
    )
    
    # Encoder Block 2: 256x256x64 -> 256x256x128 -> 128x128x128
    encoder_2_skip, encoder_2_pool = encoder_block(
        encoder_1_pool,
        num_filters=128,
        block_name='encoder_block_2',
        dropout_rate=0.4  # Add dropout for regularization
    )
    
    # ===== BOTTLENECK =====
    # Bottleneck: 128x128x128 -> 128x128x256
    bottleneck = bottleneck_block(encoder_2_pool, num_filters=256)
    
    # ===== DECODER PATH =====
    # Decoder Block 1: 128x128x256 -> 256x256x128
    decoder_1 = decoder_block(
        bottleneck,
        skip_connection=encoder_2_skip,
        num_filters=128,
        block_name='decoder_block_1'
    )
    
    # Decoder Block 2: 256x256x128 -> 512x512x64
    decoder_2 = decoder_block(
        decoder_1,
        skip_connection=encoder_1_skip,
        num_filters=64,
        block_name='decoder_block_2'
    )
    
    # ===== OUTPUT LAYER =====
    # Final convolution for segmentation - 4 classes (background + 3 tissue types)
    output_layer = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(3, 3),
        padding='same',
        activation='softmax',  # Multi-class segmentation
        name='output_segmentation'
    )(decoder_2)
    
    # Create and return the model
    model = tf.keras.Model(
        inputs=inputs, 
        outputs=output_layer,
        name='covid_lung_segmentation_unet'
    )
    
    return model
