from rimac_analytics_api.constants.constants import Constants

default_packages = Constants.default_packages


def load_packages(only_include=None, exclude=None, execute=False):
    """
    Return string with packages to load to be executed later.

    Parameters
    ----------
    only_include: list, default None.
        Group of modules to be loaded. View Constants.default_packages for choosing.
    exclude: list, default None.
        Group of modules to be loaded. View Constants.default_packages for choosing.

    Returns
    -------
    str.
    """

    def package(line):
        s = line.split(' ')[1]
        s = s[:s.find('.')] if '.' in s else s
        return s

    if only_include is None:
        only_include = list(default_packages.keys())
    if exclude is None:
        exclude = []
    packages_to_load = [item for k, sublist in default_packages.items()
                        if k in only_include and k not in exclude
                        for item in sublist]

    if execute:
        packages_to_load = [
            """
            try:
                line
            except ImportError:
                s = line.split(' ')[1]
                s = s[:s.find('.')] if '.' in s else s
                print(s)
            """
        ]

    code_to_execute = '\n'.join(packages_to_load)
    return code_to_execute

