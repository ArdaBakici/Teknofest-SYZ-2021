import tensorflow as tf
import os

def get_nr_gpu():
    """
    Returns:
        int: #available GPUs in CUDA_VISIBLE_DEVICES, or in the system.
    """
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is not None:
        return len(env.split(','))
    #logger.info("Loading devices by TensorFlow ...")
    from tensorflow.python.client import device_lib
    device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in device_protos if x.device_type == 'GPU']
    return len(gpus)

if(__name__=='__main__'):
    print (get_nr_gpu())