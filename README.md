
## Reversi Game

Welcome to our implementation of the classic Reversi game with AI opponents!


## Table of Contents
- [Game Overview](#game-overview)
- [Installation](#installation)
- [How to Play](#how-to-play)
- [Game Modes](#game-modes)
- [AI Players](#ai-players)
- [Controls](#controls)
- [Troubleshooting](#troubleshooting)

## Game Overview

Reversi is a strategic board game played on an 8×8 grid. Players take turns placing their colored discs on the board, with the goal of having the majority of their color showing when the game ends.The game is built using python.


## Requirements 
Make sure you have python installed on system. You also need to install python and required libraries before running game.


## Installation
## Installing Python (Windows/Mac/Linux):
 1. Download the latest version of Python from the  offical website :
     https://www.python.org/downloads/
 2. Run the installer and follow the instructions:
    . Windows Users : Make sure to check the box that says "ADD PYTHON TO PATH" before installing.
    . Mac/Linux Users : Python is often pre-installed, but you can verify by running python3 --version in the terminal.
  3. After installation, verify Python is installed correctly by running :
     "" python --version " or " python3 --version  ""        
##    Installation Dependencies :
 Install :"" pip install numpy "".
 ## File Extraction 
      * `reversi_FINAL.py`: Contains all the game logic and AI implementation.

### Setup
1. Extract all files from the zip folder to a directory of your choice.
   
2. Navigate to the game directory in your terminal/command prompt.

## How to Play

1. Run the main game file to start:
   
   ""python reversi_FINAL.py ""
   
2. From the start menu, select your preferred game mode and AI opponent.
3. Place your pieces by entering and make moves with rows and columns on the board.
4. The game ends when no more valid moves are available for neither player.
5. The player with the most pieces on the board wins!

## Game Modes


- **Player vs AI**: Play against an AI opponent (MCTS is used)
- **AI vs AI**: Watch two AI players compete against each other
## For AI vs AI we can configure 
   . Minimax Depth 
   . Number of MCTS simulations.

## AI Players

### Monte Carlo Tree Search (MCT)

- **Strategy**: Uses simulation-based approach to find the best moves
- **Strengths**: Good at tactical play and handling complex positions

### Minimax Algorithm

- **Strategy**: Looks ahead a certain number of moves to evaluate positions
- **Strengths**: Makes strong strategic decisions with efficient position evaluation


## Troubleshooting

### Common Issues:

1. **Game Won't Start**
   - Make sure you have Python and all required dependencies installed
   - Check that you're running the script from the correct directory
   - Verify you're using the correct filename: `reversi_FINAL.py`


2. **AI Not Working**
   - Check console output for any error messages
   
