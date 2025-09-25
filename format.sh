black . --unstable --line-length 100
isort .
find . -name '*.py' -print0 | xargs -0 pyupgrade
