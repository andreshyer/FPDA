
def parameters_to_name(parameters, suffix):
    name = ""
    for key, value in parameters.items():
        name += f"${key}={round(value, 6)}"
    if name:
        name = name[1:] + suffix
    return name


def name_to_parameters(name):
    parameters = dict()
    for key_value in name.stem.split("$"):
        key, value = key_value.split("=")
        parameters[key] = float(value)
    return parameters