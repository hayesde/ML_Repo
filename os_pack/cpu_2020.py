import os, platform, subprocess, re
from flask import jsonify

def get_os():
    my_os = platform.system()
    return my_os

def get_processor_name():
    my_os = get_os()
    if my_os == "Windows":
        my_proc = platform.processor()
    elif my_os == "Darwin":
        command = "/usr/sbin/sysctl -n machdep.cpu.brand_string"
        my_proc = subprocess.check_output(command, shell=True).strip().decode()
    elif my_os == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode()
        for line in all_info.split("\n"):
            if "model name" in line:
                my_proc = re.sub(".*model name.*:", "", line, 1)
    else:
        my_proc = "cannot find cpuinfo"
    pinfo = {"model": my_proc}
    return pinfo
~
