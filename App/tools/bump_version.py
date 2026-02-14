#!/usr/bin/env python3
"""
Usage: python3 bump_version.py [build|patch|minor|major]
Defaults to 'build' if no arg given.
This edits the 'version:' line in pubspec.yaml (e.g. 1.2.3+4).
"""
import sys, re

mode = sys.argv[1] if len(sys.argv) > 1 else "build"
path = "pubspec.yaml"

text = open(path, "r", encoding="utf-8").read()
m = re.search(r'^(?P<prefix>\s*version:\s*)(?P<ver>.+)\s*$', text, flags=re.MULTILINE)
if not m:
    print("Couldn't find a 'version:' line in pubspec.yaml")
    sys.exit(1)

ver = m.group("ver").strip().strip('"\'')
if "+" in ver:
    sem, build = ver.split("+", 1)
else:
    sem, build = ver, "0"

parts = sem.split(".")
while len(parts) < 3:
    parts.append("0")
major, minor, patch = map(int, parts[:3])
build = int(build)

if mode == "build":
    build += 1
elif mode == "patch":
    patch += 1
    build += 1
elif mode == "minor":
    minor += 1
    patch = 0
    build += 1
elif mode == "major":
    major += 1
    minor = 0
    patch = 0
    build += 1
else:
    print("Unknown mode:", mode)
    sys.exit(2)

newver = f"{major}.{minor}.{patch}+{build}"
newline = m.group("prefix") + newver
newtext = text[: m.start()] + newline + text[m.end() :]
open(path, "w", encoding="utf-8").write(newtext)
print("Updated version to", newver)
