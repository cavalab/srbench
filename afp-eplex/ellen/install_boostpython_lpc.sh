wget "https://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.tar.gz" 
tar -xzf boost_1_62_0.tar.gz 
cd boost_1_62_0
./boostrap.sh --with-libraries=python --with-python-root=/home/$USER/anaconda3
ln -s /home/$USER/anaconda/include/python3.5m /home/$USER/anaconda/include/python3.5
./b2 --with-python
