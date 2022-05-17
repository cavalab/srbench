## compile the CLI executable
make nuke && make

cp dist/regressor ${CONDA_PREFIX}/bin/gpzgd_regressor

chmod +x ${CONDA_PREFIX}/bin/gpzgd_regressor
