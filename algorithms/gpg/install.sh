#install PS Tree

# remove directory if it exists
if [ -d "gpg" ]; then
    rm -rf gpg
fi

git clone https://github.com/marcovirgolin/gpg

cd gpg
git checkout 4598833bbfbeb235af72c8f3393ae72b7eaf8d82
make
