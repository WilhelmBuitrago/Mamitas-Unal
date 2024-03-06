from tensorflow.python.client import device_lib
import tensorflow


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())

gpus = tensorflow.config.list_physical_devices('GPU')
if gpus:
    device = '/device:GPU:0'
else:
    device = '/device:CPU:0'
with tensorflow.device(device):
    print("device:", device)
