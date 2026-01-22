import numpy as np
import tensorflow as tf

print(f"NumPy Version: {np.__version__}")       # Phải là 1.23.5
print(f"TensorFlow Version: {tf.__version__}")  # Phải là 2.10.x
print("GPU Available:", len(tf.config.list_physical_devices('GPU')))