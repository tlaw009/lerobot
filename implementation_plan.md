# RA-URS LeRobot Pipeline Implementation Plan

## Goal
Integrate the custom RA-URS robotic hardware (3-DOF Arduino control + Laser) and Wolf RIWO camera into the Hugging Face LeRobot ecosystem. This unlocks native LeRobot capabilities such as teleoperation, Hugging Face dataset recording, and training state-of-the-art Neural Network policies (like Diffusion Policy and ACT).

## Open Questions
- **Gamepad Type:** Do you have an Xbox or PlayStation controller available for the lab machine? We will need to map the specific controller axes in `ra_urs_teleop.py`.
- **Operating Frequency:** How fast can the combined motor Arduino reliably receive and execute commands without desyncing? We will likely configure the LeRobot environment to run at 10 Hz or 15 Hz.

## Proposed Changes

### 1. Hardware Abstraction Layer
#### [NEW] `lerobot/ra_urs_robot.py`
*(Already Completed)* This is the LeRobot hardware wrapper. It inherits from `Robot`, initializes the serial connections to the Arduino and laser, runs a background thread with Regex to track absolute motor states, and translates absolute LeRobot actions into relative Arduino commands.

### 2. Teleoperation Interface
#### [NEW] `lerobot/ra_urs_teleop.py`
A script that bridges a game controller (via `pygame` or `inputs` library) to the `RaUrsRobot` instance.
- Maps thumbsticks to linear insertion and rotation.
- Maps D-pad or bumpers to flexion.
- Maps a trigger to the laser activation.
- Runs a continuous `while` loop at 10 Hz, computing action deltas and feeding them to `robot.send_action()`.

### 3. Dataset Recording
#### [NEW] `lerobot/ra_urs_record.py`
A data collection script that wraps the teleoperation loop.
- It will initialize LeRobot's dataset API.
- During teleoperation, it synchronizes the camera frame (`obs['observation.images.wolf_cam']`), the joint states (`obs['observation.state']`), and the human action command.
- It saves episodes sequentially to disk in the Hugging Face `.zarr` format, ready for direct training.

### 4. Neural Network Training
#### [NEW] `lerobot/configs/policy/ra_urs_diffusion.yaml`
A YAML configuration file mapping the RA-URS environment for LeRobot's Diffusion Policy architecture.
- `observation.images.wolf_cam`: Shape `[3, 800, 800]`
- `observation.state`: Dimension `4` (Linear, Rotation, Flexion, Laser)
- `action`: Dimension `4`
- You will run this phase on the server GPUs using: `python lerobot/scripts/train.py --config-name=ra_urs_diffusion`

### 5. Autonomous Deployment
#### [NEW] `lerobot/ra_urs_deploy.py`
The final inference loop.
- Loads the trained `.safetensors` policy weights.
- Grabs the live observation from `RaUrsRobot`.
- Passes the observation through the policy model to get the predicted action.
- Sends the predicted action back to the physical robot.

## Verification Plan
1. **Teleoperation Test:** Verify that the gamepad smoothly drives the hardware without throwing Arduino string-parsing errors.
2. **Dataset Verification:** Use LeRobot's visualization tools to replay a recorded `.zarr` episode and confirm the camera frames and motor states match perfectly.
3. **Training Loss:** Verify the Diffusion Policy loss decreases steadily on the server over a small subset of 10 episodes.
