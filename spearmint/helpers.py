import os
import sys
import subprocess
import tempfile

def log(*args):
    '''Write a msg to stderr.'''
    for v in args:
        sys.stderr.write(str(v))
    sys.stderr.write("\n")