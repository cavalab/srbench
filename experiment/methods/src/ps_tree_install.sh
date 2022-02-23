# install PS Tree

# remove directory if it exists
if [ -d "PS-Tree" ]; then
    rm -rf PS-Tree
fi

git clone https://github.com/Hengzhe-Zhang/PS-Tree.git

cd PS-Tree
pip install -r requirements.txt
python setup.py install