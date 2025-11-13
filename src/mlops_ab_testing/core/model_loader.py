"""
Model loading utilities for MLOps A/B Testing Framework.

This module handles loading ML models from various formats
and frameworks (scikit-learn, PyTorch, TensorFlow, etc.).
"""

import pickle
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading ML models from various formats."""
    
    SUPPORTED_FORMATS = ['.pkl', '.joblib', '.pickle', '.model', '.h5', '.pt', '.pth', '.onnx']
    
    def __init__(self):
        """Initialize ModelLoader."""
        self.loaded_models: Dict[str, Any] = {}
    
    def load_model(
        self,
        model_path: Union[str, Path],
        model_name: str,
        model_type: Optional[str] = None
    ) -> Any:
        """
        Load a model from file.
        
        Args:
            model_path: Path to model file
            model_name: Name to assign to the model
            model_type: Optional model type hint (sklearn, pytorch, tensorflow, onnx)
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is unsupported
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model '{model_name}' from {model_path}")
        
        # Determine file extension
        extension = model_path.suffix.lower()
        
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported model format: {extension}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        
        # Load based on extension or explicit type
        try:
            if model_type == 'sklearn' or extension in ['.pkl', '.pickle', '.joblib']:
                model = self._load_sklearn_model(model_path)
            elif model_type == 'pytorch' or extension in ['.pt', '.pth']:
                model = self._load_pytorch_model(model_path)
            elif model_type == 'tensorflow' or extension == '.h5':
                model = self._load_tensorflow_model(model_path)
            elif model_type == 'onnx' or extension == '.onnx':
                model = self._load_onnx_model(model_path)
            else:
                # Try pickle as default
                model = self._load_sklearn_model(model_path)
            
            # Cache the loaded model
            self.loaded_models[model_name] = model
            logger.info(f"Successfully loaded model '{model_name}'")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise
    
    def _load_sklearn_model(self, model_path: Path) -> Any:
        """
        Load scikit-learn model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded sklearn model
        """
        extension = model_path.suffix.lower()
        
        try:
            if extension == '.joblib':
                model = joblib.load(model_path)
            else:
                # Try pickle with different protocols for compatibility
                with open(model_path, 'rb') as f:
                    try:
                        model = pickle.load(f)
                    except (pickle.UnpicklingError, AttributeError, ModuleNotFoundError) as e:
                        # If pickle fails, try joblib as fallback
                        logger.warning(f"Pickle load failed, trying joblib: {str(e)}")
                        model = joblib.load(model_path)
            
            # Verify it has predict method
            if not hasattr(model, 'predict'):
                raise ValueError("Loaded object doesn't have 'predict' method")
            
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to load sklearn model: {str(e)}")
    
    def _load_pytorch_model(self, model_path: Path) -> Any:
        """
        Load PyTorch model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded PyTorch model
        """
        try:
            import torch
            
            model = torch.load(model_path, map_location='cpu')
            
            # If it's a state dict, we need the model architecture
            if isinstance(model, dict) and 'state_dict' in model:
                raise ValueError(
                    "Model file contains only state_dict. "
                    "Please provide the model architecture or full model."
                )
            
            # Set to eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
            return model
            
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch"
            )
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model: {str(e)}")
    
    def _load_tensorflow_model(self, model_path: Path) -> Any:
        """
        Load TensorFlow/Keras model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded TensorFlow model
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.models.load_model(model_path)
            
            return model
            
        except ImportError:
            raise ImportError(
                "TensorFlow not installed. Install with: pip install tensorflow"
            )
        except Exception as e:
            raise ValueError(f"Failed to load TensorFlow model: {str(e)}")
    
    def _load_onnx_model(self, model_path: Path) -> Any:
        """
        Load ONNX model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded ONNX inference session
        """
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(str(model_path))
            
            return session
            
        except ImportError:
            raise ImportError(
                "ONNX Runtime not installed. Install with: pip install onnxruntime"
            )
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {str(e)}")
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a previously loaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model object
            
        Raises:
            KeyError: If model not found
        """
        if model_name not in self.loaded_models:
            raise KeyError(f"Model '{model_name}' not loaded")
        
        return self.loaded_models[model_name]
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model '{model_name}'")
    
    def list_loaded_models(self) -> list:
        """
        Get list of loaded model names.
        
        Returns:
            List of model names
        """
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model: Model object
            
        Returns:
            Dictionary with model information
        """
        info = {
            'type': type(model).__name__,
            'module': type(model).__module__,
        }
        
        # Add scikit-learn specific info
        if hasattr(model, 'get_params'):
            info['params'] = model.get_params()
        
        # Add features info if available
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
        
        if hasattr(model, 'feature_names_in_'):
            info['feature_names'] = list(model.feature_names_in_)
        
        # Add class info for classifiers
        if hasattr(model, 'classes_'):
            info['classes'] = list(model.classes_)
            info['n_classes'] = len(model.classes_)
        
        return info


class ModelWrapper:
    """Wrapper for standardizing model inference across different frameworks."""
    
    def __init__(self, model: Any, model_type: str = 'sklearn'):
        """
        Initialize model wrapper.
        
        Args:
            model: The model object
            model_type: Type of model (sklearn, pytorch, tensorflow, onnx)
        """
        self.model = model
        self.model_type = model_type
    
    def predict(self, X: Any) -> Any:
        """
        Make predictions using the model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model_type == 'sklearn':
            return self.model.predict(X)
        
        elif self.model_type == 'pytorch':
            import torch
            import numpy as np
            
            # Convert to tensor if needed
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(np.array(X))
            
            with torch.no_grad():
                output = self.model(X)
                
                # Convert back to numpy
                if isinstance(output, torch.Tensor):
                    output = output.numpy()
                
                return output
        
        elif self.model_type == 'tensorflow':
            import numpy as np
            
            output = self.model.predict(X)
            return output
        
        elif self.model_type == 'onnx':
            import numpy as np
            
            # Get input name
            input_name = self.model.get_inputs()[0].name
            
            # Run inference
            output = self.model.run(None, {input_name: np.array(X).astype(np.float32)})
            return output[0]
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict_proba(self, X: Any) -> Any:
        """
        Get prediction probabilities (for classifiers).
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if self.model_type == 'sklearn':
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise AttributeError("Model doesn't support predict_proba")
        
        elif self.model_type == 'pytorch':
            import torch
            import numpy as np
            import torch.nn.functional as F
            
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(np.array(X))
            
            with torch.no_grad():
                output = self.model(X)
                
                # Apply softmax if needed
                if output.shape[-1] > 1:
                    proba = F.softmax(output, dim=-1)
                else:
                    proba = torch.sigmoid(output)
                
                return proba.numpy()
        
        elif self.model_type == 'tensorflow':
            output = self.model.predict(X)
            
            # Apply softmax if needed
            if len(output.shape) > 1 and output.shape[-1] > 1:
                import tensorflow as tf
                output = tf.nn.softmax(output).numpy()
            
            return output
        
        else:
            raise NotImplementedError(f"predict_proba not implemented for {self.model_type}")


def save_model(model: Any, save_path: Union[str, Path], model_type: str = 'sklearn') -> None:
    """
    Save a model to file.
    
    Args:
        model: Model object to save
        save_path: Path to save the model
        model_type: Type of model (sklearn, pytorch, tensorflow)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'sklearn':
        joblib.dump(model, save_path)
    
    elif model_type == 'pytorch':
        import torch
        torch.save(model, save_path)
    
    elif model_type == 'tensorflow':
        model.save(save_path)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Model saved to {save_path}")