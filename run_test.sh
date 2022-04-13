SUBNAME=E2ET # submission name
SUBFOLDER=submission/$SUBNAME # path containing submission
SUBENV=opensource #srcomp-$SUBNAME # the environment created for your method

# copy it over to the experiment folder
mkdir -p experiment/methods/$SUBNAME
cp -r $SUBFOLDER/* experiment/methods/$SUBNAME
touch experiment/methods/$SUBNAME/__init__.py

# run the test
cd experiment
python -m pytest -v test_submission.py --ml $SUBNAME        
