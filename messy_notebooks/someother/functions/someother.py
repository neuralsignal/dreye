

def printer(x):
    return f"""
    type: {type(x)}
    data: {x.__dict__}
    """