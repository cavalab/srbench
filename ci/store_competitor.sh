TARGET_BRANCH="official-competitors"
git config --global user.name 'GitHub Action'
git config --global user.email 'action@github.com'
git fetch
git checkout $TARGET_BRANCH
git pull origin $TARGET_BRANCH
# checkout source branch files
git checkout ${GITHUB_HEAD_REF} -- "experiment/methods/$SUBNAME/" 
# copy files from the source branch 
cp -R experiment/methods/$SUBNAME official_competitors/
# commit to the repository (ignore if no modification)
git diff-index --quiet HEAD ||  git commit -am "Store $SUBNAME as competitor"  
echo "storing competitor to $TARGET_BRANCH branch"
git push origin $TARGET_BRANCH # push to remote branch
