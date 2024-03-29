<h1 align="center">Spam filter</h1>
<p align="center">This is a small private repository made for testing machine learning</p>

## Setting up the environment
The environment is made using python 3.11, you first need to make sure it is installed on your system :
### Python 3.11 installation :
Ubuntu/Mint :
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo apt install python3-pip
```
MacOS : Download the installer here : https://www.python.org/ftp/python/3.11.7/python-3.11.7-macos11.pkg
### Creation of the environment for linux/macOS :
Open a terminal and navigate into the repositoriy directory with the cd command and launch the script with :
```
sh create_env_linux.sh
```
To activate the env use :
```
source env/bin/activate
```
while in the repository folder.

## Using the webapp
To use the webapp simply launch run_app.sh in a terminal using ```./run_app.sh``` while in the project directory.<br>
The environment is automatically created if it does not exist, you still need to have python 3.11 installed though.
### How to use the interface
The pipeline parameters can be specified in the "Parameters of the model :" field.

* alpha : (float) alpha parameter of th MLP classifier
* n_components : (int) number of components to retain for the training, will influence the training time at the cost of precision
* n_iter : (int) number of iterations to perform for the SVD, little to no influence
* max_iter : (int) maximum number of iterations to perform for the training, if set high enougth it has no influence.
* hidden_layer_sizes : (tupple(int)) shape of the MLP network
* random_state : (int) seed to use for the training, used to have consistant results

The train/test ratio specifies the relative amount of samples to used for training.<br>
The optimize button searches for optimal parameters to maximize the accuracy of the model.<br>
The train model button trains the model according to the specified settings.<br>
The submit button submits the inputed message to the model to find out wether or not it is a spam message.<br>
