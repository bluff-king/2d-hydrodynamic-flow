#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Constants and parameters
const double length_x = 1.0;
const double length_y = 0.6;
const int nx = 200;
const int ny = 120;
const double dx = length_x / nx;
const double dy = length_y / ny;
const double dt = 0.000005;
const double nu = 0.5;
const double div_tolerance = 1e-3;
const double beta0 = 1.0;
const int max_iterations = 50;
const double total_time = 0.0005;
const double beta = beta0 / (2 * dt * (1 / (dx*dx) + 1 / (dy*dy)));

// Velocity fields, pressure, and divergence
double **u; // x-velocity on cell faces (ny x nx+1)
double **v; // y-velocity on cell faces (ny+1 x nx)
double **p; // pressure at cell centers (ny x nx)
double **div_matrix; // divergence (ny x nx)

// Temporary velocity fields for updating
double **u_next;
double **v_next;

// Velocity fields at cell centers and magnitude
double **u_center; // x-velocity at cell centers (ny x nx)
double **v_center; // y-velocity at cell centers (ny x nx)
double **velocity_magnitude; // Velocity magnitude at cell centers (ny x nx)

// Obstacle mask (1 for fluid cells, 0 for obstacle cells)
int **mask; // (ny x nx)

// Obstacle definition
const int obstacle_x_start = (int)(nx * 0.3);
const int obstacle_x_end = (int)(nx * 0.4);
const int obstacle_y_start = (int)(ny * 0.1);
const int obstacle_y_end = (int)(ny * 1);

// Function to allocate memory for 2D arrays
double** allocate_2d_array(int rows, int cols) {
    double** arr = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        arr[i] = (double*)malloc(cols * sizeof(double));
        // Initialize with zeros
        for (int j = 0; j < cols; j++) {
            arr[i][j] = 0.0;
        }
    }
    return arr;
}

// Function to allocate memory for 2D integer arrays
int** allocate_2d_int_array(int rows, int cols) {
    int** arr = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        arr[i] = (int*)malloc(cols * sizeof(int));
        // Initialize with ones
        for (int j = 0; j < cols; j++) {
            arr[i][j] = 1;
        }
    }
    return arr;
}

// Function to free memory for 2D arrays
void free_2d_array(double** arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Function to free memory for 2D integer arrays
void free_2d_int_array(int** arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Function to compute the divergence in each cell
void compute_divergence() {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            div_matrix[j][i] = (u[j][i+1] - u[j][i]) / dx + (v[j+1][i] - v[j][i]) / dy;
        }
    }
}

// Function to update velocities based on Navier-Stokes equation
void update_velocities() {

    // Copy current velocities to temporary arrays
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx + 1; i++) {
            u_next[j][i] = u[j][i];
        }
    }
    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx; i++) {
            v_next[j][i] = v[j][i];
        }
    }

    // Internal cells (excluding boundaries and obstacle)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx; i++) {
            // Calculate averages for u at cell centers
            double u_i__j = 0.5 * (u[j][i-1] + u[j][i]);
            double u_i_plus1__j = 0.5 * (u[j][i] + u[j][i+1]);

            // Calculate uv product terms (Eq. uv-average)
            double uv_i_plushalf__j_minushalf = 0.25 * (u[j-1][i] + u[j][i]) * (v[j][i-1] + v[j][i]);
            double uv_i_plushalf__j_plushalf = 0.25 * (u[j][i] + u[j+1][i]) * (v[j+1][i-1] + v[j+1][i]);

            // Implement u-momentum equation (Eq. u-finite-diff)
            u_next[j][i] = u[j][i] + dt * (
                + (u_i__j*u_i__j - u_i_plus1__j*u_i_plus1__j) / dx
                + (uv_i_plushalf__j_minushalf - uv_i_plushalf__j_plushalf) / dy
                + (p[j][i-1] - p[j][i]) / dx
                + nu * (
                    (u[j][i+1] - 2*u[j][i] + u[j][i-1]) / (dx*dx)
                    + (u[j+1][i] - 2*u[j][i] + u[j-1][i]) / (dy*dy)
                )
            );
        }
    }

    for (int j = 1; j < ny; j++) {
        for (int i = 1; i < nx - 1; i++) {
            // Calculate averages for v at cell centers
            double v_i__j = 0.5 * (v[j-1][i] + v[j][i]);
            double v_i__j_plus1 = 0.5 * (v[j][i] + v[j+1][i]);

            // Calculate uv product terms (Eq. uv-average)
            double uv_i_minushalf__j_plushalf = 0.25 * (u[j-1][i] + u[j][i]) * (v[j][i-1] + v[j][i]);
            double uv_i_plushalf__j_plushalf = 0.25 * (u[j-1][i+1] + u[j][i+1]) * (v[j][i] + v[j][i+1]);

            // Implement v-momentum equation (Eq. v-finite-diff)
            v_next[j][i] = v[j][i] + dt * (
                + (v_i__j*v_i__j - v_i__j_plus1*v_i__j_plus1) / dy
                + (uv_i_minushalf__j_plushalf - uv_i_plushalf__j_plushalf) / dx
                + (p[j-1][i] - p[j][i]) / dy
                + nu * (
                    (v[j][i+1] - 2*v[j][i] + v[j][i-1]) / (dx*dx)
                    + (v[j+1][i] - 2*v[j][i] + v[j-1][i]) / (dy*dy)
                )
            );
        }
    }

    // Update the velocity fields
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx + 1; i++) {
            u[j][i] = u_next[j][i];
        }
    }
    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx; i++) {
            v[j][i] = v_next[j][i];
        }
    }

    // Apply boundary conditions
    // apply_boundary_conditions(); // Will be called after pressure correction
}

// Function to perform pressure correction iterations
void pressure_correction() {
    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute divergence
        compute_divergence();

        int done = 1;
        // Update pressure and velocity components for cells with high divergence
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (mask[j][i] == 1 && fabs(div_matrix[j][i]) > div_tolerance) {
                    done = 0;

                    double delta_p = -beta * div_matrix[j][i];
                    // Update pressure w/ pressure gradient damping
                    p[j][i] += delta_p * 0.7;

                    // Adjust velocity components (Eqs. u-update1, u-update2, v-update1, v-update2)
                    if (i + 1 <= nx) u[j][i+1] += 0.5 * dt/dx * delta_p;
                    if (i >= 0) u[j][i] -= 0.5 * dt/dx * delta_p;
                    if (j + 1 <= ny) v[j+1][i] += 0.5 * dt/dy * delta_p;
                    if (j >= 0) v[j][i] -= 0.5 * dt/dy * delta_p;
                }
            }
        }

        if (done) {
            return;
        }
    }
    // printf("\nWarning: convergence not reached\n");
}

// Function to apply boundary conditions
void apply_boundary_conditions() {
    // Left boundary (inlet): constant velocity
    for (int j = 0; j < ny; j++) {
        u[j][0] = 1.0;
    }

    // Right boundary (outlet): zero gradient
    for (int j = 0; j < ny; j++) {
        u[j][nx] = u[j][nx-1];
    }

    // Top and bottom boundaries: no-slip condition
    for (int i = 0; i < nx; i++) {
        v[0][i] = 0.0;
        v[ny][i] = 0.0;
    }
    for (int i = 0; i < nx + 1; i++) {
        u[0][i] = 0.0;
        u[ny-1][i] = 0.0;
    }

    // Obstacle boundary conditions: no-slip condition
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (mask[j][i] == 0) {
                u[j][i] = 0.0;
                u[j][i+1] = 0.0;
                v[j][i] = 0.0;
                v[j+1][i] = 0.0;
            }
        }
    }

    // Add velocity clamping
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx + 1; i++) {
            if (u[j][i] > 3.0) u[j][i] = 3.0;
            if (u[j][i] < -3.0) u[j][i] = -3.0;
        }
    }
    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx; i++) {
            if (v[j][i] > 3.0) v[j][i] = 3.0;
            if (v[j][i] < -3.0) v[j][i] = -3.0;
        }
    }
}

// Function to calculate velocities at cell centers and magnitude
void calculate_center_velocities_and_magnitude(double **u_center, double **v_center, double **velocity_magnitude) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            u_center[j][i] = 0.5 * (u[j][i] + u[j][i+1]);
            v_center[j][i] = 0.5 * (v[j][i] + v[j+1][i]);
            velocity_magnitude[j][i] = sqrt(u_center[j][i]*u_center[j][i] + v_center[j][i]*v_center[j][i]);
        }
    }
}

// Function to write the current state to the JSON file
void write_state_to_json(FILE *fp, int time_step, double current_time, double **u_center, double **v_center, double **velocity_magnitude) {
    fprintf(fp, "{\n");
    fprintf(fp, "\"time_step\": %d,\n", time_step);
    fprintf(fp, "\"current_time\": %f,\n", current_time);

    // Write u velocity matrix
    fprintf(fp, "\"u\": [\n");
    for (int j = 0; j < ny; j++) {
        fprintf(fp, "[");
        for (int i = 0; i < nx + 1; i++) {
            fprintf(fp, "%f%s", u[j][i], (i == nx) ? "" : ", ");
        }
        fprintf(fp, "]%s\n", (j == ny - 1) ? "" : ",");
    }
    fprintf(fp, "],\n");

    // Write v velocity matrix
    fprintf(fp, "\"v\": [\n");
    for (int j = 0; j < ny + 1; j++) {
        fprintf(fp, "[");
        for (int i = 0; i < nx; i++) {
            fprintf(fp, "%f%s", v[j][i], (i == nx - 1) ? "" : ", ");
        }
        fprintf(fp, "]%s\n", (j == ny) ? "" : ",");
    }
    fprintf(fp, "],\n");

    // Write u_center matrix
    fprintf(fp, "\"u_center\": [\n");
    for (int j = 0; j < ny; j++) {
        fprintf(fp, "[");
        for (int i = 0; i < nx; i++) {
            fprintf(fp, "%f%s", u_center[j][i], (i == nx - 1) ? "" : ", ");
        }
        fprintf(fp, "]%s\n", (j == ny - 1) ? "" : ",");
    }
    fprintf(fp, "],\n");

    // Write v_center matrix
    fprintf(fp, "\"v_center\": [\n");
    for (int j = 0; j < ny; j++) {
        fprintf(fp, "[");
        for (int i = 0; i < nx; i++) {
            fprintf(fp, "%f%s", v_center[j][i], (i == nx - 1) ? "" : ", ");
        }
        fprintf(fp, "]%s\n", (j == ny - 1) ? "" : ",");
    }
    fprintf(fp, "],\n");

    // Write velocity_magnitude matrix
    fprintf(fp, "\"velocity_magnitude\": [\n");
    for (int j = 0; j < ny; j++) {
        fprintf(fp, "[");
        for (int i = 0; i < nx; i++) {
            fprintf(fp, "%f%s", velocity_magnitude[j][i], (i == nx - 1) ? "" : ", ");
        }
        fprintf(fp, "]%s\n", (j == ny - 1) ? "" : ",");
    }
    fprintf(fp, "]\n");


    fprintf(fp, "}");
}


int main() {
    // Allocate memory
    u = allocate_2d_array(ny, nx + 1);
    v = allocate_2d_array(ny + 1, nx);
    p = allocate_2d_array(ny, nx);
    div_matrix = allocate_2d_array(ny, nx);
    mask = allocate_2d_int_array(ny, nx);

    u_next = allocate_2d_array(ny, nx + 1);
    v_next = allocate_2d_array(ny + 1, nx);

    u_center = allocate_2d_array(ny, nx);
    v_center = allocate_2d_array(ny, nx);
    velocity_magnitude = allocate_2d_array(ny, nx);

    // Initialize obstacle mask
    for (int j = obstacle_y_start; j < obstacle_y_end; j++) {
        for (int i = obstacle_x_start; i < obstacle_x_end; i++) {
            mask[j][i] = 0;
        }
    }

    // Initialize flow: add inlet velocity from the left
    for (int j = 0; j < ny; j++) {
        u[j][0] = 1.0;
    }

    FILE *fp = fopen("simulation_output.json", "w");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    fprintf(fp, "[\n"); // Start of JSON array

    int num_time_steps = (int)(total_time / dt);
    int output_interval = num_time_steps / 50; // Output 50 frames
    if (output_interval == 0) output_interval = 1;

    for (int t = 0; t < num_time_steps; t++) {
        update_velocities();
        pressure_correction();
        apply_boundary_conditions();
        calculate_center_velocities_and_magnitude(u_center, v_center, velocity_magnitude);

        if (t % output_interval == 0) {
            double current_time = t * dt;
            write_state_to_json(fp, t, current_time, u_center, v_center, velocity_magnitude);
            if (t < num_time_steps - output_interval) {
                 fprintf(fp, ",\n"); // Separator between states
            } else {
                 fprintf(fp, "\n"); // No separator after the last state
            }
        }
    }

    fprintf(fp, "]\n"); // End of JSON array
    fclose(fp);

    // Free memory
    free_2d_array(u, ny);
    free_2d_array(v, ny + 1);
    free_2d_array(p, ny);
    free_2d_array(div_matrix, ny);
    free_2d_int_array(mask, ny);
    free_2d_array(u_next, ny);
    free_2d_array(v_next, ny + 1);
    free_2d_array(u_center, ny);
    free_2d_array(v_center, ny);
    free_2d_array(velocity_magnitude, ny);
    printf("Simulation completed.\n");

    return 0;
}
