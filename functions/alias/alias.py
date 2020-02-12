"""
Format:
@aliased
class MyClass(object):
    @alias('name1', 'name2')
    def some_method(self):
    ...

c = MyClass
c.name1()

Taken from http://code.activestate.com/recipes/577659-decorators-for-adding-aliases-to-methods-in-a-clas/
Written by: José Nahuel
"""


class Alias(object):

    def __init__(self, *aliases):
        self.aliases = set(aliases)

    def __call__(self, f):
        f._aliases = self.aliases
        return f


def aliased(aliased_class):
    original_methods = aliased_class.__dict__.copy()
    for name, method in original_methods.items():
        if hasattr(method, '_aliases'):
            for alias in method._aliases - set(original_methods):
                setattr(aliased_class, alias, method)
    return aliased_class
