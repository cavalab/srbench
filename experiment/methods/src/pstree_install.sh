#install PS Tree

# remove directory if it exists
if [ -d "PS_Tree" ]; then
    rm -rf PS_Tree
fi

git clone https://github.com/zhenlingcn/PS-Tree

cd PS_Tree
python setup.py install
