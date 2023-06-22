import threading
from typing import Union

from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone

class SharedInformation:
    def __init__(self):
        self.choice_inputs = []
        self.choice_outputs = []

        self.queue_inputs = []
        self.queue_outputs = []
        self._queue_lock = threading.Lock()

        self.model_trained: DecisionTreeRegressor = DecisionTreeRegressor()
        self.model_outdated: bool = False
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

    def copy_decision_tree(self, model_trained):
        with self._model_lock:
            self.model_trained = clone(model_trained)
            self.model_outdated = True

    def get_decision_tree(self) -> tuple[bool, Union[None, DecisionTreeRegressor]]:
        with self._model_lock:
            if not self.model_outdated:
                return False, None
            return True, self.model_trained
