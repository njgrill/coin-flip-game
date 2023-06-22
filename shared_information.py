import threading
from sklearn import tree

class SharedInformation:
    def __init__(self):
        self.choice_inputs = []
        self.choice_outputs = []

        self.queue_inputs = []
        self.queue_outputs = []
        self._queue_lock = threading.Lock()

        self.model_training = None
        self.model_trained = None
        self._model_lock = threading.Lock()

    def add_to_queue(self, input, output):
        with self._queue_lock:
            self.queue_inputs.append(input)
            self.queue_outputs.append(output)

    def copy_queue(self):
        with self._queue_lock:
            self.choice_inputs.append(self.queue_inputs)
            self.choice_outputs.append(self.queue_outputs)
            self.queue_inputs = []
            self.queue_outputs = []

    def copy_decision_tree(self):
        with self._model_lock:

