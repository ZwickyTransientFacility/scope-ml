# For use by pip-installed scope package
from scope.scope_class import Scope

scope = Scope()


def develop():
    scope.develop()


def doc():
    scope.doc()


def train():
    scope.parse_run_train()


def create_training_script():
    scope.parse_run_create_training_script()


def assemble_training_stats():
    scope.parse_run_assemble_training_stats()


def create_inference_script():
    scope.parse_run_create_inference_script()


def select_fritz_sample():
    scope.parse_run_select_fritz_sample()


def test_limited():
    scope.test_limited()


def test():
    scope.parse_run_test()
