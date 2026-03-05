import tensorflow as tf

class HardwarePerformanceCallback(tf.keras.callbacks.Callback):
    def __init__(self, logger):
        super(HardwarePerformanceCallback, self).__init__()
        self.logger = logger
    
    def on_epoch_begin(self, epoch, logs=None):
        self.logger.record_epoch_start()
    
    def on_epoch_end(self, epoch, logs=None):
        self.logger.record_epoch_end()