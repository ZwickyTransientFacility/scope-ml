# For use by pip-installed scope package
from scope.scope import Scope

scope = Scope()


def develop():
    scope.develop()


# def lint()
# class method
#    Scope.lint()


def test_limited():
    scope.test_limited()


def test():
    scope.test()


# def...
