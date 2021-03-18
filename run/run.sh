pip install git+https://github.com/lucasb-eyer/pydensecrf.git
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/opencv_python_headless.libs:$LD_LIBRARY_PATH
pip install opencv-python-headless==3.4.*
ldconfig
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/opencv_python_headless.libs:$LD_LIBRARY_PATH
pip install git+https://github.com/minouei-kl/cral
python run.py
