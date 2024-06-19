import os
import math

MAX_ROTATE = 99
SUFFIX_LEN = int(math.log10(MAX_ROTATE)) + 1


def rotate(filename, num_rotate=MAX_ROTATE):
    if num_rotate > MAX_ROTATE:
        raise RuntimeError(f"num_rotate too large! max {MAX_ROTATE}, got {num_rotate}")

    filename = os.path.normcase(os.path.normpath(filename))

    # rotate
    suffix = str(num_rotate).rjust(SUFFIX_LEN, "0")
    rotated_filename = filename + "." + suffix
    if os.path.exists(rotated_filename):
        os.remove(rotated_filename)
    for rotate_id in range(num_rotate - 1, 0, -1):
        suffix = str(rotate_id).rjust(SUFFIX_LEN, "0")
        rotated_filename = filename + "." + suffix
        if os.path.exists(rotated_filename):
            new_suffix = str(rotate_id + 1).rjust(SUFFIX_LEN, "0")
            new_filename = filename + "." + new_suffix
            os.rename(rotated_filename, new_filename)
    if os.path.exists(filename):
        suffix = str(1).rjust(SUFFIX_LEN, "0")
        os.rename(filename, filename + "." + suffix)
