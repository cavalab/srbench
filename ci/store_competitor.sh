git config --global user.name 'GitHub Action'
git config --global user.email 'action@github.com'
git add experiment/methods/$SUBNAME/
git diff-index --quiet HEAD ||  git commit -am "Store $SUBNAME as competitor"  # commit to the repository (ignore if no modification)
echo "pushing to ${GITHUB_HEAD_REF}"
git push origin ${GITHUB_HEAD_REF} # push to remote branch
