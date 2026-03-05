from tensorflow import keras
from app.common.socketCommunication import *

class EndEpoch(keras.callbacks.Callback):
    #def __init__(self, socket:SocketCommunication):
    #    self.socket = socket

    def on_epoch_end(self, batch, logs={}):
        SocketCommunication.decide_print_form(MSGType.END_EPOCH, 
            {'node': 2, 'msg': 'Slave ended epoch', 'loss': logs.get('loss'),
                'val_loss': logs.get('val_loss'), 'accuracy': logs.get('accuracy'), 
                'val_accuracy': logs.get('val_accuracy')})