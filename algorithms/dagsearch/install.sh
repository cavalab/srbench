#install dag search

# remove directory if it exists
if [ -d "DAG_search" ]; then
    rm -rf DAG_search
fi

git clone https://github.com/kahlmeyer94/DAG_search

cd DAG_search
pip install -r requirements.txt
python setup.py install