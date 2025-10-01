black . --unstable --line-length 120
isort . --line-length 120
find . -name '*.py' -print0 | xargs -0 pyupgrade
autoflake --in-place --recursive .

