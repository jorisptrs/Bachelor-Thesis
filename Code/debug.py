from env_variables import DEBUG

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)