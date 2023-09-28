mkdir model_paths
mkdir images
mkdir tb_logs
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install git+https://github.com/treforevans/uci_datasets.git
pip install matplotlib
pip install scikit_learn==1.3.0
pip install pandas==2.0.3
pip install tqdm==4.66.1
pip install opencv_python==4.8.0.76
pip install ConfigArgParse==1.7
pip install pytorch_lightning==1.9.5
pip install lightning_bolts==0.7.0
pip install gspread==5.11.2
pip install oauth2client==4.1.3
pip install seaborn==0.12.2
pip install pyinterval==1.2.0
pip install jaxlib==0.4.14
pip install jax==0.4.14
pip install pymc3==3.11.5