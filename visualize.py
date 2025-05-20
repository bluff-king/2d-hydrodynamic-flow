import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from IPython.display import HTML
import os

dtype_map = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64
}

def load_metadata(file_path):
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    return metadata

# Function to read raw binary data to 3d np array (num_frames, rows, cols)
def read_binary_data(filename, rows, cols, dtype_str):

    np_dtype = dtype_map.get(dtype_str)
    if np_dtype is None:
        raise Exception(f"Error: Unsupported data type '{dtype_str}' in metadata")

    if not os.path.exists(filename):
        raise Exception(f"Error: File not found: {filename}")


    with open(filename, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np_dtype)

    expected_elements_per_frame = rows * cols
    num_frames_in_file = arr.size // expected_elements_per_frame

    assert arr.size % expected_elements_per_frame == 0, 'Number of elements are not divisible by rows*cols'

    reshaped_arr = arr.reshape((num_frames_in_file, rows, cols))
    return reshaped_arr

# Function to load simulation data from binary files
def load_simulation_data(output_dir, metadata):
    nx = metadata['nx']
    ny = metadata['ny']
    data_dtype = metadata['data_dtype']

    dims = {
        'u_center': (ny, nx),
        'v_center': (ny, nx),
        'velocity_magnitude': (ny, nx)
    }

    loaded_data = {}
    
    for var_name, (rows, cols) in dims.items():
        filename = os.path.join(output_dir, f"{var_name}_data.bin")
        loaded_data[var_name] = read_binary_data(filename, rows, cols, data_dtype)
        if loaded_data[var_name] is None:
            raise Exception(f"Failed to load {var_name}")

    num_frames = metadata['num_frames_output']
    
    simulation_data_list = []
    for i in range(num_frames):
        time_step = i * metadata['output_interval_in_c_steps']
        current_time = time_step * metadata['dt']

        frame_data = {
            'u_center': loaded_data['u_center'][i],
            'v_center': loaded_data['v_center'][i],
            'velocity_magnitude': loaded_data['velocity_magnitude'][i],
            'time_step': time_step,
            'current_time': current_time
        }
        simulation_data_list.append(frame_data)
        
    return simulation_data_list

# Function to create the animation
def create_animation(simulation_data, metadata):
    nx = metadata['nx']
    ny = metadata['ny']
    length_x = metadata['length_x']
    length_y = metadata['length_y']
    # obstacle_x_start = metadata['obstacle_x_start']
    # obstacle_x_end = metadata['obstacle_x_end']
    # obstacle_y_start = metadata['obstacle_y_start']
    # obstacle_y_end = metadata['obstacle_y_end']

    fig, ax = plt.subplots(figsize=(10, 6))

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
        u_center = state['u_center']
        v_center = state['v_center']
        velocity_magnitude = state['velocity_magnitude']
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

def main():
    output_dir = ""
    metadata_file = os.path.join(output_dir, "simulation_metadata.json")

    metadata = load_metadata(metadata_file)
    simulation_data = load_simulation_data(output_dir, metadata)

    anim = create_animation(simulation_data, metadata)
    # To display in Jupyter Notebook, the HTML object needs to be the last line
    # HTML(anim.to_jshtml()) # This line should be uncommented in the notebook

    # For saving the animation to a file
    anim.save(
        'simulation_animation.gif',
        writer='ffmpeg',
        fps=5
    )


if __name__ == '__main__':
    main()