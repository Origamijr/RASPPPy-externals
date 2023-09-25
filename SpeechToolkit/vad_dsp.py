import torch
import os
import numpy as np
from scipy.signal import resample
from math import ceil
import onnxruntime as ort
ort.set_default_logger_severity(3)

from raspppy.core.object import AsyncObject, IOType
from raspppy.core.config import config
from raspppy.core.logger import log

class VAD_DSP(AsyncObject):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_input(IOType.SIGNAL)
        self.add_output()
        self.add_output()

        self.sr = 16000
        self.win_size = 512
        
        self.properties = {
            'speech_threshold': 0.5,
            'silence_threshold': 0.35,
            'min_speech_duration': 0.1,
            'min_silence_duration': 0.3,
        } | self.properties

        self.leftover_audio = None
        self.audio_queue = []
        self.in_speech_arr = []
        self.in_silence_arr = []
        self.in_speech = False

        self.ready = False
        self._spawn(self._get_model)

    def _get_model(self):
        path = os.path.join(config('files', 'model_dir'), 'silero')
        if not os.path.exists(path):
            log("downloading VAD model...")
            os.makedirs(path)
            torch.hub.set_dir(path)
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True,
                                      onnx=True)
        else:
            for dir, _, files in os.walk(path):
                if 'hubconf.py' in files:
                    path = dir
                    break
            model, _ = torch.hub.load(repo_or_dir=path,
                                      model='silero_vad',
                                      force_reload=True,
                                      onnx=True,
                                      source='local')
        self.model = model
        self.ready = True
    
    def process_signal(self):
        if not self.ready: return

        runtime_sr = config('audio', 'sample_rate')
        needed_samples = ceil(self.win_size * runtime_sr / self.sr)
        audio = torch.tensor(self.inputs[0].value, dtype=torch.float32)

        # prepend leftover audio from last block
        if self.leftover_audio is not None: audio = torch.concatenate((self.leftover_audio, audio))

        # detect speech on as many frames as possible in current block
        while len(audio) >= needed_samples:
            frame = torch.tensor(resample(audio[:needed_samples], self.win_size))
            audio = audio[needed_samples:]
            prob = self.model(frame, self.sr).item()
            self.in_speech_arr.append(prob > self.properties['speech_threshold'])
            self.in_silence_arr.append(prob < self.properties['silence_threshold'])
            self.audio_queue.append(frame)
            
            min_speech_duration = ceil(self.properties['min_speech_duration'] * self.sr / self.win_size)
            min_silence_duration = ceil(self.properties['min_silence_duration']  * self.sr / self.win_size)
            
            # if not in speech and speech is detected, enter speech state
            if not self.in_speech and len(self.in_speech_arr) >= min_speech_duration and all(self.in_speech_arr[-min_speech_duration:]):
                self.in_speech = True
                self.outputs[1].value = 1
                self.send(port=1)
            
            elif len(self.in_speech_arr) >= min_silence_duration and all(self.in_silence_arr[-min_silence_duration:]):
                # if in speech, but now detects silence, store audio and return to silence state
                if self.in_speech: 
                    self.in_speech = False
                    self.outputs[1].value = 0
                    self.send(port=1)
                    
                    self.outputs[0].value = np.concatenate((self.audio_queue))
                    self.send(port=0)
                
                # reset queues (since we don't care about silence data)
                self.in_speech_arr = self.in_speech_arr[-min_silence_duration:]
                self.in_silence_arr = self.in_silence_arr[-min_silence_duration:]
                self.audio_queue = self.audio_queue[-min_silence_duration:]

        self.leftover_audio = audio