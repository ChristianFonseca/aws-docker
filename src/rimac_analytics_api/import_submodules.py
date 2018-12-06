import_submodules = \
"""import pkgutil

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if is_pkg and module_name.find('.') < 0:
        __all__.append(module_name)
    module = loader.find_module(module_name).load_module(module_name)
    exec('%s = module' % module_name)

del loader, module_name, is_pkg, module, pkgutil"""