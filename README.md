# PPO for Super Mario Bros 🍄🤖

This project implements the Proximal Policy Optimization (PPO) algorithm with an Actor-Critic architecture to train an AI agent to play Super Mario Bros. The agent learns to navigate the game environment by processing visual input (frames from the game) and receiving rewards.

## Project Overview 🌟

The primary goal is to create an autonomous agent capable of achieving high scores and completing levels in Super Mario Bros. The implementation uses:
-   **TensorFlow/Keras** 🧠 for building and training the neural network models.
-   **OpenAI Gym** and **gym-super-mario-bros** 🎮 for the game environment.
-   **PPO Algorithm** 📈 for stable and efficient policy updates.
-   **Actor-Critic Architecture** 🎭 where the Actor decides the action and the Critic evaluates the state.
-   **CNN (Convolutional Neural Network)** 🖼️ to process game frames.
-   Techniques like frame stacking, grayscale conversion, and image resizing to preprocess observations.
-   Parallel environment interaction using multiple "actors" 🏃‍♂️💨 to gather diverse experiences.

## Key Features ✨

*   ✅ Proximal Policy Optimization (PPO)
*   ✅ Actor-Critic Neural Network Model
*   ✅ Convolutional Neural Network (CNN) for visual input processing
*   ✅ Frame Stacking for temporal information
*   ✅ Grayscale and Resized Image Observations for efficiency
*   ✅ Parallel data collection with multiple game environments (actors)
*   ✅ Model saving and loading capabilities 💾
*   ✅ Separate modes for training a new model and running a pre-trained model

## Prerequisites 💻

*   **Python 3.9** 🐍 (This is a strict requirement as per instructions)
*   `pip` (Python package installer)

## Installation ⚙️

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/omerjakoby/MARIO-RL-PPO.git
    cd MARIO-RL-PPO
    ```

2.  **Install the required Python libraries:**
    Make sure you are in the project's root directory (`MARIO-RL-PPO`) where `requirements.txt` is located.
    ```bash
    pip install -r requirements.txt
    ```

## Usage ▶️

The script `ppo_mario.py` handles both training new models and running pre-trained ones.

### 1. Preparing Pre-trained Models (Optional)  pretrained

If you have pre-trained `actor` and `critic` models (e.g., named `actor_model_v550` and `critic_model_v550`):

*   Ensure these model directories are present in the project's root directory or a known location.
*   By default, the script tries to load models from paths like `r"actor_model_v550"` and `r"critic_model_v550"`.
*   **If the script cannot find the model directories ❗:**
    You will need to provide the **absolute paths** to your `actor_model_v550` and `critic_model_v550` directories. Modify lines 256 and 257 in `ppo_mario.py`:
    ```python
    # In ppo_mario.py around line 256:
    actor = keras.models.load_model(r"C:\path\to\your\actor_model_v550") # Replace with your absolute actor path
    critic = keras.models.load_model(r"C:\path\to\your\critic_model_v550") # Replace with your absolute critic path
    ```

### 2. Training a New Model from Scratch 🛠️

If you want to train models from scratch:

*   **Comment out** the model loading lines (around 256-257) in `ppo_mario.py`:
    ```python
    # actor = keras.models.load_model(r"actor_model_v550")
    # critic = keras.models.load_model(r"critic_model_v550")
    ```
*   When you run the script, you will choose option `2` to start training.

### 3. Running the Script 🚀

Execute the main Python script:

```bash
python ppo_mario.py
