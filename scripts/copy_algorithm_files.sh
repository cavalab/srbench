SUBNAME=$1
SUBFOLDER="./algorithms/$SUBNAME"
echo "Copying files and environment to experiment/methods ..."
echo "........................................"
mkdir -p experiment/methods/$SUBNAME
cp $SUBFOLDER/regressor.py experiment/methods/$SUBNAME/
cp $SUBFOLDER/metadata.yml experiment/methods/$SUBNAME/
touch experiment/methods/$SUBNAME/__init__.py