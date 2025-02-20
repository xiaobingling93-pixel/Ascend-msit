SHELL_DIR=$(dirname $(readlink -f $0))
python3 -m coverage run --branch --source ${SHELL_DIR}/../ms_service_profiler_ext -m pytest ${SHELL_DIR}/ut
python3 -m coverage report -m