import whisper
import torch
import numpy as np
import os

from raspppy.core.object import AsyncObject
from raspppy.core.config import config

class Whisper(AsyncObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_input()
        self.add_output()
        
        self.properties = {
            'model_name': 'base.en',
            'cuda': True
        } | self.properties

        self.ready = False
        self._spawn(self._get_model)
        
    def _get_model(self):
        path = os.path.join(config('files', 'model_dir'), 'whisper')
        file = self.properties['model_name']
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, file+'.pt')):
            self.model = whisper.load_model(file, download_root=path)
        else:
            self.model = whisper.load_model(os.path.join(path, file + '.pt'))

        # send a "dummy message to warm it up"
        self.model.transcribe(np.array([0] * (1*512), dtype=np.float32), fp16=(self.properties['cuda'] and torch.cuda.is_available()))

        self.ready = True

    def bang(self, port=0):
        if not self.ready: return
        if not isinstance(self.inputs[0].value, np.ndarray): return
        text = self.model.transcribe(self.inputs[0].value, fp16=(self.properties['cuda'] and torch.cuda.is_available()))['text'].strip()
        self.outputs[0].value = text
        self.send()

if __name__ == "__main__":
    import time
    Whisper()
    time.sleep(3)