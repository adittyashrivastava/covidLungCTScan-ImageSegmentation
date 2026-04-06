"""Model factory for loading different segmentation architectures."""

from .unet import build_unet
from .segnet import build_segnet
from .deeplabv3plus import build_deeplabv3plus


# Registry of available models
MODEL_REGISTRY = {
    'unet': build_unet,
    'segnet': build_segnet,
    'deeplabv3plus': build_deeplabv3plus,
}


def get_model(model_name, input_shape=(512, 512, 1), num_classes=4):
    """Factory function to get a segmentation model by name.
    
    Args:
        model_name: Name of the model ('unet', 'segnet', 'deeplabv3plus')
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of segmentation classes
    
    Returns:
        tf.keras.Model: The requested model
    
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_builder = MODEL_REGISTRY[model_name]
    return model_builder(input_shape=input_shape, num_classes=num_classes)


def list_available_models():
    """Get list of available model names."""
    return list(MODEL_REGISTRY.keys())