black . --unstable --line-length 120
isort .
find . -name '*.py' -print0 | xargs -0 pyupgrade
