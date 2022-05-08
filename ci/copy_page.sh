if [ -v SRC_DIR ] ; then
    echo "SRC_DIR exists:$SRC_DIR"
else
    SRC_DIR=$(pwd)
    echo "SRC_DIR not found, setting to $SRC_DIR"
fi
echo "copying $SRC_DIR/$SRC_FILE to $TARGET_DIR/$SRC_FILE"
git config --global user.name 'GitHub Action'
git config --global user.email 'action@github.com'
git fetch                         # fetch branches
git checkout $TARGET_BRANCH       # checkout to your branch
git pull origin $TARGET_BRANCH
git checkout ${GITHUB_REF##*/} -- "$SRC_DIR/$SRC_FILE" # copy files from the source branch
mkdir -p $TARGET_DIR
mv "$SRC_DIR/$SRC_FILE" $TARGET_DIR
file_path="${TARGET_DIR}/${SRC_FILE}"
git add $file_path
git diff-index --quiet HEAD ||  git commit -am "deploy updates to ${SRC_FILE}"  # commit to the repository (ignore if no modification)
git push origin $TARGET_BRANCH # push to remote branch
