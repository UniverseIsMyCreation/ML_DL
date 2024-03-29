set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk

    pip3 install -qq \
        catboost==1.0.6 \
        gdown==4.6.4 \
        h5py==3.7.0 \
        hyperopt==0.2.7 \
        ipympl==0.9.3 \
        ipywidgets==8.0.2 \
        keras==2.11.0 \
        lightgbm==3.3.2 \
        matplotlib-inline==0.1.6 \
        matplotlib==3.5.3 \
        numpy==1.23.4 \
        pandas==1.5.2 \
        pep8==1.7.1 \
        plotly==5.6.0 \
        pytest==7.1.3 \
        scikit-image==0.19 \
        scikit-learn==1.1.3 \
        scipy==1.9.3 \
        seaborn==0.12.0 \
        tensorflow==2.11.0 \
        torch==1.12.1 \
        torchvision==0.13.1 \
        tqdm==4.64.1 \
        umap-learn==0.5.3 \
        xgboost==1.6.2 \
        pycodestyle==2.9.1
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"