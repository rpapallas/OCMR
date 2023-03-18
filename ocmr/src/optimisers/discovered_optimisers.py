import pkgutil
import pathlib

all_optimisers = dict()


def register_optimiser(class_to_register):
    all_optimisers[class_to_register.__name__] = class_to_register


current_directory = pathlib.Path(__file__).parent
for loader, module_name, is_module in pkgutil.iter_modules([str(current_directory)]):
    if module_name.startswith('optimiser_'):
        loader.find_module(module_name).load_module(module_name)

