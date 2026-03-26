# battleship-ai

# Introduction
This is an AI that is trained to play the board game "Battleship." This utilizes a reinforcement learning agent as well as a Q-table approach and move-masking logic.

# Installation
Before you begin, ensure that you are running the latest version of Python. Anything from 3.7 onward is sufficient.

To get started, run the following command to install all requirements.
```bash
pip install -r requirements.txt
```
This will install the necessary Python libraries or confirm that they are already installed.

# Training
To train the AI model, go onto your terminal and enter the following command.
```bash
python train_battleship.py
```
This will train the model and generate the model.pkl file that saves all the games played during the training process. Take note that this could anywhere from 10-20 minutes. Once done, a message will display and the model.pkl file will appear in your repo.

# Playing
To begin the game, go onto your terminal and enter the following command.
```bash
python play_battleship.py
```
From there, the AI will hide the ships based on its knowledge of the game. Your goal is to find where the AI hid the ships. To do that, enter the coordinates of the ships in row-column format (ex. entering "2 2" will fire at row 2, column 2).

The board will show an H if a ship is hit, and an M if the shot is a miss. There are five total ships - 1 each of lengths 5, 4 and 2, and 2 each of length 3.

# Troubleshooting
If you run into problems while trying to run the script, try these fixes:

### 1. Python/Pip Command Not Found
This depends on how Python was installed. Depending on that, you can try one of these fixes:
- **Try: ** 'python train_battleship.py' OR
- **Try: ** 'pip3 train_battleship.py'

### 2. Numpy Not Found
This could mean that the library was installed in the wrong environment.
- **Try: ** 'pip install -r requirements.txt' OR
- **Try: ** 'pip install numpy'

### 3. FileNotFound Error
This can occur if you try running play_battleship.py before running train_battleship.py. This is because train_battleship.py generates a model.pkl file that acts as the AI's brain.
- **Fix: ** Run 'train_battleship.py' first before running 'play_battleship.py'

### 4. Display Issues
This can occur if the terminal looks off or messy.
- **Fix: ** Adjust the size of the terminal or the font size.

### 5. Slow Training
This can occur due to the number of episodes played when running train_battleship.py
- **Fix: ** You can adjust the number of episodes within 'train_battleship.py' by reducing them directly in the file. This however will create a less refined model for the AI to work with.