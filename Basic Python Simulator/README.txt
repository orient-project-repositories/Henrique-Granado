----------------packages required------------------
numpy
numpy-quaternion
gym
pyglet
numba (for better performance)

----------------------Usage-------------------------
- Running sys_playground.py:
    -Description: This script allows you to visualize the motor-eye system and play with the motor orientations, as well as the time step of the simulation
    -Usage:
            Running this script will open up 2 windows: a graphics window and a application window.

            In the application window there are 2 sections: one at the top which will allow you to control the time step, by moving the scroll left and right; and the one at the bottom which will allow to control the rotation (in degrees) of each of the motor by moving the scrollers left and right as well. To update the rotation of the motors click on the button "Get update", or check the box "Autoupdate" to automatically update every time one of the scrollers is moved.

            The graphics window works as follow:
                - By clicking on it and pressing "e" on the keyboard will lock and unlock the mouse to window.
                - When the mouse is locked:
                    * Press "w", "a", "s", "d" on the keyboard to move the player forward, to the left, backwards, or to the right respectively
                    * Press "space" to move the player upwards
                    * Press "shift" to move the player downwards
                    * Press "i" to print to the console the position and orientation of the player
                    * Press "j" to print to the console the orientation of the eye in quaternions
                    * Press "esc" to exit the program (might cause errors)


- Creating an environment:
    To create an environment of the system and apply any reinforcement learning algorithm to it, a new class inherited from the CustomEnv class should be created.
    This CustomEnv class has as its state variable which is a quaternion array that describes the orientation of the eye in the model and takes as input in the step method an action which is an array of 3 values corresponding to the rotation of each of the 3 motors.
    This CustomEnv class has also editable methods called inside the step function:
        - reward_function: in this method you define and return the reward for each time step
        - is_done: this method return whether or not the condition for the episode to be done is met. As it is setup, the condition is if a certain time as passed
        - process_action: sometimes the action passed to the the step method can't be in the form of an array with 3 values ready to be fed into the system - there might the need to have access to internal variables of the environment. This method takes as inputs such action value and can be used to process this correctly in order to be fed correctly into the simulator.
