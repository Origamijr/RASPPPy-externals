from llama_cpp import Llama as LLM

from raspppy.core.object import AsyncObject
from raspppy.core.utils import filter_kwargs

class Llama(AsyncObject):
    """
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_input()
        self.add_output()
        if 'model_path' not in self.properties:
            self.properties['model_path'] = ''
            self.model = None
        else:
            self._spawn(self.load_model)

    def load_model(self):
        self.model = LLM(**filter_kwargs(self.properties, exclude=None, adapt_f=LLM))

    def set_properties(self, *args, **kwargs):
        super().set_properties(*args, **kwargs)
        if 'model_path' in kwargs:
            self._spawn(self.load_model)

    def bang(self, port=0):
        if port == 0:
            input = self.inputs[0].value if isinstance(self.inputs[0].value, str) else str(self.inputs[0].value)
            output = self.model(input, **filter_kwargs(self.properties, exclude=None, adapt_f=self.model))
            self.outputs[0].value = output['choices'][0]['text']
            self.send()
