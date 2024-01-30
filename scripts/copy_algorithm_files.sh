SUBNAME=$(basename $1)
SUBFOLDER="$(dirname $1)/${SUBNAME}"
if (($#>1)); #check if number of arguments is 1 
then
    DEST=$2
else
    DEST="experiment/methods"
fi
echo "Copying files and environment to experiment/methods ..."
echo "........................................"
mkdir -p $DEST/$SUBNAME
cp $SUBFOLDER/regressor.py $DEST/$SUBNAME/
cp $SUBFOLDER/metadata.yml $DEST/$SUBNAME/
touch $DEST/$SUBNAME/__init__.py
