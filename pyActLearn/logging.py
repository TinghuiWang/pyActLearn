def logging_name(obj):
    """Get full path of the class

    Args:
        obj (:obj:`class`): A class object
    """
    return obj.__class__.__module__ + '.' + obj.__class__.__name__
