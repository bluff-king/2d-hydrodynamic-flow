#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Constants and parameters
const double length_x = 1.0;
const double length_y = 0.6;
const int nx = 200;
const int ny = 120;
const double dx = length_x / nx;
const double dy = length_y / ny;
const double dt = 0.000001;
const double nu = 0.5;
const double div_tolerance = 1e-3;
const double beta0 = 1.0;
const int max_iterations = 200;
const double total_time = 0.0005;
const double beta = beta0 / (2 * dt * (1 / (dx * dx) + 1 / (dy * dy)));

const int max_number_of_frames = 50;

// Velocity fields, pressure, and divergence
double **u;           // x-velocity on cell faces (ny x nx+1)
double **v;           // y-velocity on cell faces (ny+1 x nx)
double **p;           // pressure at cell centers (ny x nx)
double **div_matrix;  // divergence (ny x nx)

// Temporary velocity fields for updating
double **u_next;
double **v_next;

// Velocity fields at cell centers and magnitude
double **u_center;            // x-velocity at cell centers (ny x nx)
double **v_center;            // y-velocity at cell centers (ny x nx)
double **velocity_magnitude;  // Velocity magnitude at cell centers (ny x nx)

// Obstacle mask (1 for fluid cells, 0 for obstacle cells)
int **mask;  // (ny x nx)

// Obstacle definition
const int obstacle_x_start = (int)(nx * 0.3);
const int obstacle_x_end = (int)(nx * 0.4);
const int obstacle_y_start = (int)(ny * 0);
const int obstacle_y_end = (int)(ny * 0.9);

// Function to convert float to half-precision float
uint16_t float_to_half(float f) {
    // Convert float to bits
    uint32_t f_bits = *((uint32_t *)&f);

    // Extract sign, mantissa and exponent
    uint16_t sign = (f_bits >> 31) & 0x01;
    uint16_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mant = f_bits & 0x7FFFFF;

    // Handle special cases
    if (f == 0.0f) return 0;
    if (exp == 0xFF && mant == 0) return (sign << 15) | 0x7C00;  // Infinity
    if (exp == 0xFF && mant != 0) return (sign << 15) | 0x7E00;  // NaN

    // Calculate exponent for half precision
    int16_t half_exp = exp - 127 + 15;

    // Handle normal numbers
    if (half_exp >= 1 && half_exp <= 30) {
        // Normal number
        mant = mant >> 13;
        return (sign << 15) | (half_exp << 10) | mant;
    } else if (half_exp <= 0 && half_exp >= -10) {
        // Subnormal number (denormalized)
        // Shift mantissa to account for the exponent difference
        // Include the implied leading 1 bit for the mantissa
        mant = (mant | 0x800000) >> (14 - half_exp);
        return (sign << 15) | mant;
    } else if (half_exp > 30) {
        // Overflow to infinity
        return (sign << 15) | 0x7C00;
    } else {
        // Underflow to zero
        return (sign << 15);
    }
}

// Function to allocate memory for 2D arrays
double **allocate_2d_array(int rows, int cols) {
    double **arr = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        arr[i] = (double *)malloc(cols * sizeof(double));
        // Initialize with zeros
        for (int j = 0; j < cols; j++) {
            arr[i][j] = 0.0;
        }
    }
    return arr;
}

// Function to allocate memory for 2D integer arrays
int **allocate_2d_int_array(int rows, int cols) {
    int **arr = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        arr[i] = (int *)malloc(cols * sizeof(int));
        // Initialize with ones
        for (int j = 0; j < cols; j++) {
            arr[i][j] = 1;
        }
    }
    return arr;
}

// Function to free memory for 2D arrays
void free_2d_array(double **arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Function to free memory for 2D integer arrays
void free_2d_int_array(int **arr, int rows) {
    for (int i = 0; i < rows; i++) {
        free(arr[i]);
    }
    free(arr);
}

// Function to apply boundary conditions
void apply_boundary_conditions() {
    // Left boundary (inlet): constant velocity
    for (int j = 0; j < ny; j++) {
        u[j][0] = 1.0;
    }

    // Right boundary (outlet): zero gradient
    for (int j = 0; j < ny; j++) {
        u[j][nx] = u[j][nx - 1];
    }

    // Top and bottom boundaries: no-slip condition
    for (int i = 0; i < nx; i++) {
        v[0][i] = 0.0;
        v[ny][i] = 0.0;
    }
    for (int i = 0; i < nx + 1; i++) {
        u[0][i] = 0.0;
        u[ny - 1][i] = 0.0;
    }

    // Obstacle boundary conditions: no-slip condition
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (mask[j][i] == 0) {
                u[j][i] = 0.0;
                u[j][i + 1] = 0.0;
                v[j][i] = 0.0;
                v[j + 1][i] = 0.0;
            }
        }
    }

    // Add velocity clamping
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx + 1; i++) {
            if (u[j][i] > 5.0) u[j][i] = 5.0;
            if (u[j][i] < -5.0) u[j][i] = -5.0;
        }
    }
    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx; i++) {
            if (v[j][i] > 5.0) v[j][i] = 5.0;
            if (v[j][i] < -5.0) v[j][i] = -5.0;
        }
    }
}

// Function to compute the divergence in each cell
void compute_divergence() {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            div_matrix[j][i] =
                (u[j][i + 1] - u[j][i]) / dx + (v[j + 1][i] - v[j][i]) / dy;
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
            double u_i__j = 0.5 * (u[j][i - 1] + u[j][i]);
            double u_i_plus1__j = 0.5 * (u[j][i] + u[j][i + 1]);

            // Calculate uv product terms (Eq. uv-average)
            double uv_i_plushalf__j_minushalf =
                0.25 * (u[j - 1][i] + u[j][i]) * (v[j][i - 1] + v[j][i]);
            double uv_i_plushalf__j_plushalf = 0.25 * (u[j][i] + u[j + 1][i]) *
                                               (v[j + 1][i - 1] + v[j + 1][i]);

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
            double v_i__j = 0.5 * (v[j - 1][i] + v[j][i]);
            double v_i__j_plus1 = 0.5 * (v[j][i] + v[j + 1][i]);

            // Calculate uv product terms (Eq. uv-average)
            double uv_i_minushalf__j_plushalf =
                0.25 * (u[j - 1][i] + u[j][i]) * (v[j][i - 1] + v[j][i]);
            double uv_i_plushalf__j_plushalf = 0.25 *
                                               (u[j - 1][i + 1] + u[j][i + 1]) *
                                               (v[j][i] + v[j][i + 1]);

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
        // Update pressure and velocity components for cells with high
        // divergence
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (mask[j][i] == 1 && fabs(div_matrix[j][i]) > div_tolerance) {
                    done = 0;

                    double delta_p = -beta * div_matrix[j][i];
                    // Update pressure w/ pressure gradient damping
                    p[j][i] += delta_p * 0.7;

                    // Adjust velocity components (Eqs. u-update1, u-update2,
                    // v-update1, v-update2)
                    if (i + 1 <= nx) u[j][i + 1] += 0.5 * dt / dx * delta_p;
                    if (i >= 0) u[j][i] -= 0.5 * dt / dx * delta_p;
                    if (j + 1 <= ny) v[j + 1][i] += 0.5 * dt / dy * delta_p;
                    if (j >= 0) v[j][i] -= 0.5 * dt / dy * delta_p;
                }
            }
        }

        if (done) {
            return;
        }
    }
    // printf("\nWarning: convergence not reached\n");
}

// Function to calculate velocities at cell centers and magnitude
void calculate_center_velocities_and_magnitude() {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            u_center[j][i] = 0.5 * (u[j][i] + u[j][i + 1]);
            v_center[j][i] = 0.5 * (v[j][i] + v[j + 1][i]);
            velocity_magnitude[j][i] = sqrt(u_center[j][i] * u_center[j][i] +
                                            v_center[j][i] * v_center[j][i]);
        }
    }
}

// Function to write the current state to the binary file
void write_state_to_binary(FILE *fp_u_center, FILE *fp_v_center,
                           FILE *fp_magnitude) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            uint16_t half_val = float_to_half((float)u_center[j][i]);
            fwrite(&half_val, sizeof(uint16_t), 1, fp_u_center);
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            uint16_t half_val = float_to_half((float)v_center[j][i]);
            fwrite(&half_val, sizeof(uint16_t), 1, fp_v_center);
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            uint16_t half_val = float_to_half((float)velocity_magnitude[j][i]);
            fwrite(&half_val, sizeof(uint16_t), 1, fp_magnitude);
        }
    }
}

// Function to write metadata to a JSON file
void write_metadata_to_json(const char *filename, double compute_time_seconds) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "    \"length_x\": %f,\n", length_x);
    fprintf(fp, "    \"length_y\": %f,\n", length_y);
    fprintf(fp, "    \"nx\": %d,\n", nx);
    fprintf(fp, "    \"ny\": %d,\n", ny);
    fprintf(fp, "    \"dx\": %f,\n", dx);
    fprintf(fp, "    \"dy\": %f,\n", dy);
    fprintf(fp, "    \"dt\": %f,\n", dt);
    fprintf(fp, "    \"nu\": %f,\n", nu);
    fprintf(fp, "    \"div_tolerance\": %e,\n", div_tolerance);
    fprintf(fp, "    \"beta0\": %f,\n", beta0);
    fprintf(fp, "    \"max_iterations\": %d,\n", max_iterations);
    fprintf(fp, "    \"total_time\": %f,\n", total_time);
    fprintf(fp, "    \"beta\": %f,\n", beta);
    fprintf(fp, "    \"obstacle_x_start\": %d,\n", obstacle_x_start);
    fprintf(fp, "    \"obstacle_x_end\": %d,\n", obstacle_x_end);
    fprintf(fp, "    \"obstacle_y_start\": %d,\n", obstacle_y_start);
    fprintf(fp, "    \"obstacle_y_end\": %d,\n", obstacle_y_end);

    fprintf(fp, "    \"data_dtype\": \"float16\",\n");
    fprintf(fp, "    \"output_interval_in_c_steps\": %d,\n",
            (int)(total_time / dt) / max_number_of_frames);
    fprintf(fp, "    \"num_frames_output\": %d,\n",
            ((int)(total_time / dt) /
             ((int)(total_time / dt) / max_number_of_frames)));
    fprintf(fp, "    \"total_compute_time_seconds\": %f\n",
            compute_time_seconds);

    fprintf(fp, "}");
    fclose(fp);
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

    FILE *fp_u_center = fopen("u_center_data.bin", "wb");
    FILE *fp_v_center = fopen("v_center_data.bin", "wb");
    FILE *fp_magnitude = fopen("velocity_magnitude_data.bin", "wb");

    if (fp_u_center == NULL || fp_v_center == NULL || fp_magnitude == NULL) {
        perror("Error opening binary files");
        return 1;
    }

    int num_time_steps = (int)(total_time / dt);
    int output_interval = num_time_steps / max_number_of_frames;
    if (output_interval == 0) output_interval = 1;

    double total_compute_time_seconds = 0;

    // Computation starts here
    for (int t = 0; t < num_time_steps; t++) {
        clock_t start_time = clock();
        update_velocities();
        pressure_correction();
        apply_boundary_conditions();
        calculate_center_velocities_and_magnitude(u_center, v_center,
                                                  velocity_magnitude);
        clock_t end_time = clock();
        total_compute_time_seconds +=
            (double)(end_time - start_time) / CLOCKS_PER_SEC;

        if (t % output_interval == 0) {
            write_state_to_binary(fp_u_center, fp_v_center, fp_magnitude);
        }
    }

    // Close binary files
    fclose(fp_u_center);
    fclose(fp_v_center);
    fclose(fp_magnitude);

    write_metadata_to_json("simulation_metadata.json",
                           total_compute_time_seconds);

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
