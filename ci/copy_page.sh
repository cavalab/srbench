git config --global user.name 'GitHub Action'
git config --global user.email 'action@github.com'
git fetch                         # fetch branches
git checkout $TARGET_BRANCH       # checkout to your branch
git pull origin $TARGET_BRANCH
git checkout ${GITHUB_REF##*/} -- $SRC_FILE # copy files from the source branch
mkdir -p $TARGET_DIR
mv $SRC_FILE $TARGET_DIR
file_path="${TARGET_DIR}/${SRC_FILE}"
git add $file_path
git diff-index --quiet HEAD ||  git commit -am "deploy updates to ${SRC_FILE}"  # commit to the repository (ignore if no modification)
git push origin $TARGET_BRANCH # push to remote branch
