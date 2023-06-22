import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"GPU devices found {len(gpu_devices)}\n")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
