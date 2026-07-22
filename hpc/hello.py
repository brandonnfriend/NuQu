#!/usr/bin/env python3
# hello.py — writes proof that our Python ran on the compute node.
import datetime
import platform
import sys

msg = (
    f"hello from {platform.node()} | "
    f"python {sys.version.split()[0]} | "
    f"{datetime.datetime.now().isoformat()}"
)
print(msg)
with open("hello_out.txt", "w") as fh:
    fh.write(msg + "\n")
