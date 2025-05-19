import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import re
import ast

# Function to extract constant values from C code
def get_constant(c_code_string: str) -> dict:
    constants_dict = {}
    target_variables = ['length_x', 'length_y', 'nx', 'ny']


    variable_pattern = re.compile(
        r'^\s*const\s+\S+\s+({})'.format('|'.join(target_variables)) +
        r'\s*=\s*(.*?);'
    )

    lines = c_code_string.splitlines()

    for line in lines:
        match = variable_pattern.search(line)
        if match:
            variable_name = match.group(1)
            value_string = match.group(2).strip()
            evaluated_value = ast.literal_eval(value_string)
            constants_dict[variable_name] = evaluated_value


    return constants_dict

# Load and parse the JSON data
def load_simulation_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to create the animation
def create_animation(simulation_data, nx, ny, length_x, length_y):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the obstacle boundaries
    obstacle_x_start = int(nx * 0.3)
    obstacle_x_end = int(nx * 0.4)
    obstacle_y_start = int(ny * 0.3)
    obstacle_y_end = int(ny * 0.7)

    # Create a mask for the obstacle (1 for fluid cells, 0 for obstacle cells)
    mask = np.ones((ny, nx))
    # mask[obstacle_y_start:obstacle_y_end, obstacle_x_start:obstacle_x_end] = 0

    # Create a meshgrid for plotting
    x = np.linspace(0, length_x, nx)
    y = np.linspace(0, length_y, ny)
    X, Y = np.meshgrid(x, y)

    # Function to update each frame
    def update(frame):
        ax.clear()

        state = simulation_data[frame]
        u = np.array(state['u'])
        v = np.array(state['v'])
        u_center = np.array(state['u_center'])
        v_center = np.array(state['v_center'])
        velocity_magnitude = np.array(state['velocity_magnitude'])
        time_step = state['time_step']
        current_time = state['current_time']

        # Create a masked version for plotting
        masked_magnitude = np.ma.masked_array(velocity_magnitude, mask=1-mask)

        # Plot velocity magnitude
        contour = ax.pcolormesh(X, Y, masked_magnitude, cmap='hot', shading='gouraud')

        # Use a sparser grid for the streamplot to avoid clutter
        stride = 1
        ax.streamplot(X[::stride, ::stride], Y[::stride, ::stride],
                      u_center[::stride, ::stride], v_center[::stride, ::stride],
                      color='blue', density=1, linewidth=0.5, arrowsize=0.5)

        # Set fixed axis limits to prevent glitching
        ax.set_xlim(0, length_x)
        ax.set_ylim(0, length_y)
        
        # Force aspect ratio to be equal
        ax.set_aspect('equal')

        ax.set_title(f'Flow simulation (Time Step: {time_step}, Time: {current_time:.5f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Use tight layout but with padding to prevent cutting off labels
        plt.tight_layout(pad=1.0)

        return contour,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(simulation_data), interval=200, blit=False)

    return anim

# Main execution block for Jupyter Notebook
def main(source_code: str):

    name = source_code[:-2]
    json_file_path = f'{name}_output.json'
    simulation_data = load_simulation_data(json_file_path)

    with open(source_code, 'r') as f:
        c_code = f.read()
    
    const_dict = get_constant(c_code)

    nx = const_dict['nx']
    ny = const_dict['ny']
    length_x = const_dict['length_x']
    length_y = const_dict['length_y']

    anim = create_animation(simulation_data, nx, ny, length_x, length_y)
    # To display in Jupyter Notebook, the HTML object needs to be the last line
    # HTML(anim.to_jshtml()) # This line should be uncommented in the notebook

    # For saving the animation to a file (optional)
    anim.save('simulation_animation.gif', writer='pillow', fps=5)


if __name__ == '__main__':
    main('simulation.c')