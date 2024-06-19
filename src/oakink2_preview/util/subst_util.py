import re

match_special = re.compile(r"^\?\((.*)\)$")


def extract_special(s):
    match_special_res = match_special.fullmatch(s)
    if match_special_res:
        return match_special_res.group(1)
    else:
        return None


match_special_part = re.compile(r"\?\((.*?)\)")


def extract_special_part(s):
    cmd_list = []
    span_list = []
    for match in match_special_part.finditer(s):
        cmd = match.group(1)
        span = match.span()
        cmd_list.append(cmd)
        span_list.append(span)
    return cmd_list, span_list


def replace_from_span(s, span_list, replacement_list):
    # s is a string, span_list contains index of start and end of each span in the *original string*, replacement_list contains replacement string
    # return a new string with replacement
    assert len(span_list) == len(replacement_list)
    # sort by start index
    span_list = sorted(span_list, key=lambda x: x[0])
    # replace from end to start
    for i in range(len(span_list) - 1, -1, -1):
        start, end = span_list[i]
        s = s[:start] + replacement_list[i] + s[end:]
    return s


match_file = re.compile(r"^file:(.*)$")


def extract_file(s):
    match_file_res = match_file.fullmatch(s)
    if match_file_res:
        return match_file_res.group(1)
    else:
        return None
