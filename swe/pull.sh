#!/bin/zsh
# Pulls the 16 task images (x86_64, run under emulation on Apple Silicon).
cd ${0:A:h}
for id in $(python3 -c "import json;print(' '.join(json.load(open('tasks16.json'))))"); do
  img="docker.io/swebench/sweb.eval.x86_64.${id//__/_1776_}:latest"
  docker pull --platform linux/amd64 -q $img > /dev/null && echo "ok $id" || echo "FAIL $id"
done
