SUBNAME=$1
SUBENV=srcomp-$SUBNAME

if (($#==1)); #check if number of arguments is 1 
then
    subnames=($1)
else
    # subnames=$(ls -d official_competitors/*/)
    # subnames=$(cd official_competitors; ls -d *; cd ..)
    subnames=(
    Bingo
    E2ET
    HROCH # failing
    PS-Tree
    QLattice
    TaylorGP
    eql
    geneticengine
    gpzgd
    nsga-dcgp
    operon
    pysr
    uDSR
    )
fi
echo "subnames:$subnames"

cd experiment

for SUBNAME in ${subnames[@]} ; do 
    SUBENV=srcomp-$SUBNAME
    echo "activating conda env $SUBENV..."
    echo "........................................"
    eval "$(conda shell.bash hook)"
    conda init bash
    conda activate $SUBENV
    conda env list 
    conda info 

    # Test Method
    ls
    python -m pytest -vv test_submission.py --ml $SUBNAME
done
