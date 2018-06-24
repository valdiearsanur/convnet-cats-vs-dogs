# Convolutional Neural Network
This project using two kind of Architecture. First is VGG-16 and second is my custom minimal architecture (the alternative when you want to train on your machine).

## 1. Training
### 1A. Train with floydHub
You'll save your time with train this stuff on the cloud. First you will have to signed up on FloydHub (https://www.floydhub.com) and create a project. Then run a job on FloydHub by simply command below:
```
floyd init [your project]
floyd run \
    --data sominw/datasets/dogsvscats/1:workspace/dog-vs-cat \
    --tensorboard "python workspace/dog-vs-cat/classification.py --logdir /output/Graph" 
```
#### After process is completed
1. Your weight data is saved on `/workspace/dog-vs-cat/output/weights.hdf5`. You will need that file for making prediction

### 1B. Train with Jupyter Notebook in your machine (using Docker)
With this step, you have to install Docker in your machine. In docker machine, I use helpful docker container https://github.com/ufoym/deepo to to get my machine set up with all Deep Learning tools. Run this docker command to set your machine
```
docker run -it -p 8888:8888 \
    -v "$PWD"/../workspace:/workspace \
    --ipc=host \
    ufoym/deepo:py36-jupyter-cpu \
    jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/workspace'
```

Your machine will set up with Jupyter Notebook (with ML tools).

#### Next Steps :
1. when 0.0.0.0:8000 is ready, open classificatin file in http://0.0.0.0:8888/notebooks/dog-vs-cat/classification.ipynb
1. Click on Nav menu `Kernel` > `Restart & Run All`, your codes will run
1. Your weight data is saved on `/workspace/dog-vs-cat/output/weights.hdf5`. You will need that file for making prediction