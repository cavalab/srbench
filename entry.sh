#!/bin/bash
## Entrypoint for docker RUN directives as well as ENTRYPOINT with conda env
## Enable by adding:
##   COPY entry.sh ./
##   SHELL ["/entry.sh", "/bin/bash", "-c"]
##
## Optionally, set the following env to select a conda env to run in
##   ENV CONDA_DEFAULT_ENV=foo
## You may also want to add something like
##   RUN conda init bash && echo 'conda activate "${CONDA_TARGET_ENV:-base}"' >>  ~/.bashrc
## to drop into a default env when `docker exec -it $IMAGE bash`
## Docker shells by default run as nonlogin, noninteractive
## More references:
##  https://pythonspeed.com/articles/activate-conda-dockerfile/
##  https://stackoverflow.com/questions/56510575/activate-and-switch-anaconda-environment-in-dockerfile-during-build
##  https://stackoverflow.com/questions/37945759/condas-source-activate-virtualenv-does-not-work-within-dockerfile/62803490#62803490

## It is insufficient to run `conda init bash` in a dockerfile, and then `source $HOME/.bashrc` in the entry script.
## This is mostly because the `RUN` directives are noninteractive, non-login shells, meaning `.bashrc` is never 
## sourced, and `RUN source` does not behave the way one might naively think it should
## However, by taking the `conda shell.bash hook` directly, we end up with a conda-tized
## RUN directive!
## The conda shell hook placed in `.bashrc` will reset our
## env to "base" on shell-ing into the container. If you want to start in a custom end,


# cache the value because the shell hook step will remove it
_CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-base}"

__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

# Restore our "indended" default env
conda activate "${_CONDA_DEFAULT_ENV}"
# This just logs the output to stderr for debugging. 
>&2 echo "ENTRYPOINT: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

exec "$@"