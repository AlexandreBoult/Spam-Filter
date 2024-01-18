<h1 style="text-align: center">Spam filter</h1>
<p style="text-align: center">This is a small private repository made for testing machine learning</p>

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
To use the webapp simply launch run_app.sh in a terminal using ```./run_app.sh``` while in the project directory.
You need to set up the environment before running the app.
### How to use the interface
The pipeline parameters can be specified in the "Parameters of the model :" field.

