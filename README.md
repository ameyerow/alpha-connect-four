# Alpha Zero in Connect Four

AlphaZero is a state-of-the-art DL/RL algorithm developed by the DeepMind team, capable of playing Go at superhuman levels. In this project, we implement a much smaller version of AlphaZero while preserving its basic structure, including MCTS and policy-value network. Our implementation achieved near optimal play against a random policy baseline in Connect Two, and ~75\% winrate in Connect 4. However, the model is limited by available compute and particularly the speed of MCTS, which slows training data generation and limits model complexity.

# Prerequisites
Python versions and dependencies are managed with pipenv. If you do not have pipenv already,
install and verify the installation.
```
pip install pipenv
pipenv -h
```

# Running
When running the project for the first time, it is necessary to install dependencies. Run the 
following commands from the root directory to properly set up and activate your environment.
```
pipenv install
pipenv shell
```
On subsequent runs, the python enviornment will already have been created via pipenv so you can simply activate it and run the desired file, i.e. `leaner.py`.
```
pipenv shell
python -m src.learner
```

# Playing

To play against a trained model, active the environment and run `play.py`.
```
pipenv shell
python -m src.play
```
By default this plays Connect4, but this can be switched to the test environment "Connect2".
```
python -m src.play --connect2
```
Highlighted blue circles in the environment represent legal action, click on them to make your move. If there are no highlighted circles this means its the model's turn. The speed of the model's turns depends on your computer's hardware.

# Training

To train model, active the environment and run `learner.py`.
```
pipenv shell
python -m src.learner
```
There are several flags available in the `learner.py` file, run the following to get a list of them.
```
python -m src.learner -h
```
The code supports CUDA devices as long as a CUDA-enabled version of pytorch is installed. The correct pip install command for your hardware can be found on pytorch's [website](https://pytorch.org/get-started/locally/). With the pipenv environment active simply paste their command into the terminal and execute it. After the installation, training should make use of any CUDA enabled device in your system.