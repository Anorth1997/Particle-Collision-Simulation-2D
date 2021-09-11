/* ------------
 * The code is adapted from the XSEDE online course Applications of Parallel Computing. 
 * The copyright belongs to all the XSEDE and the University of California Berkeley staff
 * that moderate this online course as well as the University of Toronto CSC367 staff.
 * This code is provided solely for the use of students taking the CSC367 course at 
 * the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * -------------
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "common.h"

//  tuned constants
#define cutoff  0.01
#define density 0.0005

enum ParticleRegionStatus {
    LOCAL, // 0, particles in the local region
    GHOST_ABOVE, // 1, particles in the above ghost zone
    GHOST_BELOW, // 2, particles in the below ghost zone
    NEITHER // 3, particles outside
};

enum MPI_TAG {
    RECEIVE_ABOVE, // exchange particles in the region with above
    RECEIVE_ABOVE_GHOST, // exchange particles in the ghost with above
    RECEIVE_BELOW, // exchange particles in the region with below
    RECEIVE_BELOW_GHOST // exchange particles in the ghost with below
};

struct node {
    int index;

    node *next;
};

// Each grid is a linked_list
class Grid
{
public:
    node *head;

    Grid()
    {
        head = NULL;
    }

    // add the node to the end of the linked_list
    void add_node(int index)
    {
        node *tmp = new node;
        tmp->index = index;
        tmp->next = NULL;

        if(head == NULL)
        {
            head = tmp;
        }
        else
        {
            tmp->next = head;
            head = tmp;
        }
    }

    // clean the list for next timestep (clean the heap)
    void clean_list()
    {
        node *cur_node = head;
        while(cur_node) {
            node *pre_node = cur_node;
            cur_node = cur_node->next;
            delete pre_node;
        }
        head = NULL;
    }

    // given node that represents the particle, compute its collision with all other particles in this cell
    void handle_particle_given_cell(node *cur_node, particle_t *particles_to_move, particle_t *particles_to_compare, double *dmin, double *davg, int *navg) {
        node *tmp = head;
        while (tmp) {
            apply_force(particles_to_move[cur_node->index], particles_to_compare[tmp->index], dmin,davg,navg);
            tmp = tmp->next;
        }
    }

    // compute all particles in this cell
    void compute_all_particles_in_cell(Grid *grids, particle_t *local_particles, particle_t *above_ghost_particles, particle_t * below_ghost_particles,
                                       int above_ghost_zone_exist, int below_ghost_zone_exist,
                                       int x, int y, int length, int height, double *dmin, double *davg, int *navg) {
        node *cur_node = head;

        while (cur_node) {
            // give a particle, compute it with nearby grids
            for (int i = -1; i <= 1; i++) { // iterate 3 times
                for (int j = -1; j <= 1; j++) { // iterate 3 times
                    int fixed_x = x + i;
                    int fixed_y = y + j;
                    if (fixed_x >= 0 && fixed_x < height && fixed_y >= 0 && fixed_y < length) { // valid grid
                        Grid grid_to_interact = grids[fixed_x * length + fixed_y];
                        // given a particle and the a grid, apply force
                        // if the fixed_x and fixed_y points to a grid in region, we pass the local_particles
                        if (above_ghost_zone_exist && fixed_x == 0) { // fixed_x and fixed_y points above ghost zone
                            grid_to_interact.handle_particle_given_cell(cur_node, local_particles, above_ghost_particles, dmin,davg,navg);
                        } else if (below_ghost_zone_exist && fixed_x == height - 1) { // fixed_x and fixed_y points below ghost zone
                            grid_to_interact.handle_particle_given_cell(cur_node, local_particles, below_ghost_particles, dmin,davg,navg);
                        } else { // grid in region
                            grid_to_interact.handle_particle_given_cell(cur_node, local_particles, local_particles, dmin,davg,navg);
                        }
                    }
                }
            }
            cur_node = cur_node->next;
        }
    }

    int size()
    {
        int size = 0;
        node *cur_node = head;
        while(cur_node) {
            size++;
            cur_node = cur_node->next;
        }
        return size;
    }
};

ParticleRegionStatus decide_particle_status(double x, double x_min, double x_max, int rank, int n_proc, double grid_width) {
    if (rank == 0) { // an extra row of grids below
        if (x >= x_min && x < x_max) {
            return LOCAL;
        } else if (x >= x_max && x < x_max + grid_width) {
            return GHOST_BELOW;
        } else {
            return NEITHER;
        }
    } else if (rank == n_proc - 1) { // an extra row of grids above
        if (x >= x_min && x < x_max) {
            return LOCAL;
        }
        else if (x >= x_min - grid_width && x < x_min) {
            return GHOST_ABOVE;
        } else {
            return NEITHER;
        }
    } else { // an extra row of grids below and an extra row of grids above
        if (x >= x_min && x < x_max) {
            return LOCAL;
        }
        else if (x >= x_max && x < x_max + grid_width) {
            return GHOST_BELOW;
        }
        else if (x >= x_min - grid_width && x < x_min) {
            return GHOST_ABOVE;
        }
        else {
            return NEITHER;
        }
    }
}

// helper function to compute if a particle is in the prcocess's region
// this function take care of the case n_proc = 1
int is_particle_in_region(double x, double x_min, double x_max) {
    return x >= x_min && x < x_max;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg;

    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    MPI_Request send_above_request, send_above_ghost_request, send_below_request, send_below_ghost_request;
    MPI_Request receive_above_request, receive_above_ghost_request, receive_below_request, receive_below_ghost_request;

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );

    // ------------------------- My solution -------------------------------
    /* 1. Horizontally partition the whole map based on the number of processes.
     *
     * Each process manage its own local variable for map boundary
     * CORRECT
     */
    // size of the whole map along a side
    double size_t;
    size_t = sqrt( density * n );

    // each process has its own x_min and x_max recording the boundary of the its region
    double region_height = size_t / n_proc;

    // every particle has x >= x_min and x < x_max
    double x_min = rank * region_height;
    double x_max = (rank + 1) * region_height;


    /* 2. Initialize all the particles
     *
     * Let the master process initialize all the particles at once, and distribute the
     * particles' index to corresponding destination process.
     * CORRECT
     */

    // all the particles
    particle_t *all_particles = (particle_t*) malloc( n * sizeof(particle_t) );

    MPI_Datatype PARTICLE;  /* initialize particle datatype */
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );  /* 6 doubles in particle_t */
    MPI_Type_commit( &PARTICLE ); /* Commit the datatype so we can use it in communications */

    if (rank == 0) {
        // master process init all the particles and broadcast them
        init_particles( n, all_particles );
    }
    MPI_Bcast(all_particles, n, PARTICLE, 0, MPI_COMM_WORLD);


    /* 3. Send particles to corresponding process based on particle's location
     *
     * Each process's manages the particles it has in "local_particles"
     */
    double grid_width = cutoff;

    particle_t local_particles[n];
    particle_t above_ghost_particles[n];
    particle_t below_ghost_particles[n];

    int local_particles_num = 0;
    int above_ghost_num = 0;
    int below_ghost_num = 0;

    for (int i = 0; i < n; i++) {
        ParticleRegionStatus status = decide_particle_status(all_particles[i].x, x_min, x_max, rank, n_proc, grid_width);
        if (status == LOCAL) {
            local_particles[local_particles_num] = all_particles[i];
            local_particles_num += 1;
        }
        else if (status == GHOST_ABOVE) {
            above_ghost_particles[above_ghost_num] = all_particles[i];
            above_ghost_num += 1;
        } else if (status == GHOST_BELOW) {
            below_ghost_particles[below_ghost_num] = all_particles[i];
            below_ghost_num += 1;
        }
    }

    /* 4. Each process init local grids including Ghost zone
      *
      * for process rank == 0:
      * the last row of grids is the ghost zone
      * grids with indices i >= (height - 1) * length and i < height * length
      *
      * for process rank == n_proc - 1:
      * the first row of grids is the ghost zone
      * grids with indices i >= 0 and i < length
      *
      * for process in between:
      * the first row of grids and last row of grids together is the ghost zone
      * grids with indices i >= 0 and i < length; i >= (height - 1) * length and i < height * length
      *
      */
    // number of grids horizontally
    int length = ceil(size_t / grid_width);
    // number of grids vertically
    int height = ceil(region_height / grid_width);

    if (rank == 0 || rank == n_proc - 1) {
        height += 1;
    } else {
        height += 2;
    }

    // initialize the grids (including the ghost zone)
    Grid grids[height * length];


    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
            if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, all_particles );


        for (int i = 0; i < local_particles_num; i++) {
            local_particles[i].ax = local_particles[i].ay = 0;
        }

        //
        // 1. Allocate local_particles and ghost_particles to grid (including the ghost zone)
        // CORRECT
        for (int i = 0; i < local_particles_num; i++) {
            // compute which grid this particle belongs to, and insert the nodes
            int row_index = (local_particles[i].x - x_min) / grid_width;
            if (rank > 0) { // fix the grid index because of ghost zone
                row_index += 1;
            }
            int col_index = local_particles[i].y / grid_width;
            int grid_index = row_index * length + col_index;

            // add the node to its grid, i represents the index of local_particles array.
            grids[grid_index].add_node(i);
        }

        // the last process won't even enter this loop
        for (int i = 0; i < below_ghost_num; i++) {
            int row_index = height - 1;
            int col_index = below_ghost_particles[i].y / grid_width;
            int grid_index = row_index * length + col_index;
            grids[grid_index].add_node(i);
        }

        // the first process won't even enter this loop
        for (int i = 0; i < above_ghost_num; i++) {
            int row_index = 0;
            int col_index = above_ghost_particles[i].y / grid_width;
            int grid_index = row_index * length + col_index;
            grids[grid_index].add_node(i);
        }

        //
        // 2. do collide computation except the ones in the ghost zone
        //
        int start_grid_index;
        int end_grid_index;
        if (rank == 0) {
            start_grid_index = 0;
            end_grid_index = (height - 1) * length;
        } else if (rank == n_proc - 1) {
            start_grid_index = length;
            end_grid_index = height * length;
        } else { // middle processes
            start_grid_index = length;
            end_grid_index = (height - 1) * length;
        }

        int above_ghost_zone_exist = 0;
        int below_ghost_zone_exist = 0;
        for (int i = start_grid_index; i < end_grid_index; i++) {
            int x = i / length;
            int y = i % length;

            // mark the ghost zone
            if (n_proc > 1) {
                if (rank == 0) {
                    below_ghost_zone_exist = 1;
                } else if (rank == n_proc - 1) {
                    above_ghost_zone_exist = 1;
                } else {
                    below_ghost_zone_exist = 1;
                    above_ghost_zone_exist = 1;
                }
            }

            // i = x * length + y
            // handle all the particles in this grid
            grids[i].compute_all_particles_in_cell(grids, local_particles, above_ghost_particles, below_ghost_particles,
                                                   above_ghost_zone_exist, below_ghost_zone_exist, x, y, length, height, &dmin,&davg,&navg);
        }

        //
        // 3. free the nodes in every grid
        //
        for (int i = 0; i < height * length; i++) {
            grids[i].clean_list();
        }


        //
        // 4. move particles, except the ones in the ghost zone
        //
        for (int i = 0; i < local_particles_num; i++) {
            move(local_particles[i]);
        }


        /*
         * Process exchange information
         *
         * we need to update the following information
         * local particles
         * above ghost particles
         * below ghost particles
         */

        //
        // 5. send the leaving local particles and Receive particles
        //
        // iterate through the particles out of boundary and send them out
        if (n_proc > 1) {
            if (rank == 0) { // top region
                // number of particles that are still in the region except the ones in ghost zone
                int remaining_particles_num = 0;
                // number of particles that are out of the region
                int send_below_particles_num = 0;
                // number of particles resides in the neighbor's region
                int send_below_ghost_particles_num = 0;

                particle_t tmp_local_particles[local_particles_num];
                particle_t send_below_buffer[local_particles_num];
                particle_t send_below_ghost_particles[local_particles_num];

                for (int i = 0; i < local_particles_num; i++) {
                    if (is_particle_in_region(local_particles[i].x, x_min, x_max)) { // particles in the region (ghost zone excluded)
                        tmp_local_particles[remaining_particles_num] = local_particles[i];
                        remaining_particles_num += 1;
                    } else { // the particles that leave the region
                        send_below_buffer[send_below_particles_num] = local_particles[i];
                        send_below_particles_num += 1;
                    }

                    if (local_particles[i].x < x_max && local_particles[i].x >= x_max - grid_width) {
                        send_below_ghost_particles[send_below_ghost_particles_num] = local_particles[i];
                        send_below_ghost_particles_num += 1;
                    }
                }


                // collect the particles that still exist in the region
                memcpy(local_particles, tmp_local_particles, remaining_particles_num * sizeof(particle_t));

                // update the number of local particles
                local_particles_num = remaining_particles_num;

                MPI_Status receive_below_status, receive_below_ghost_status;

                particle_t receive_from_below_buffer[n - remaining_particles_num];


                // receive the new particles coming into the region
                MPI_Irecv(receive_from_below_buffer,
                          n - remaining_particles_num,
                          PARTICLE,
                          rank + 1,
                          RECEIVE_BELOW,
                          MPI_COMM_WORLD,
                          &receive_below_request);

                // send the particles going out side the region
                MPI_Isend(send_below_buffer,
                          send_below_particles_num,
                          PARTICLE,
                          rank + 1,
                          RECEIVE_ABOVE,
                          MPI_COMM_WORLD,
                          &send_below_request);

                // receive the new ghost zone particles from neighbor process
                MPI_Irecv(below_ghost_particles,
                          n,
                          PARTICLE,
                          rank + 1,
                          RECEIVE_BELOW_GHOST,
                          MPI_COMM_WORLD,
                          &receive_below_ghost_request);

                // send the new ghost zone particles for neighbor process
                MPI_Isend(send_below_ghost_particles,
                          send_below_ghost_particles_num,
                          PARTICLE,
                          rank + 1,
                          RECEIVE_ABOVE_GHOST,
                          MPI_COMM_WORLD,
                          &send_below_ghost_request);

                MPI_Wait(&receive_below_request, &receive_below_status);
                MPI_Wait(&send_below_request, MPI_STATUS_IGNORE);
                MPI_Wait(&receive_below_ghost_request, &receive_below_ghost_status);
                MPI_Wait(&send_below_ghost_request, MPI_STATUS_IGNORE);

                // get the number of received particles in region
                int num_incoming_region_particles;
                MPI_Get_count(&receive_below_status, PARTICLE, &num_incoming_region_particles);

                // get the number of received particles in ghost
                int num_incoming_below_ghost_particles;
                MPI_Get_count(&receive_below_ghost_status, PARTICLE, &num_incoming_below_ghost_particles);

                memcpy(local_particles + remaining_particles_num, receive_from_below_buffer, num_incoming_region_particles * sizeof(particle_t));
                local_particles_num += num_incoming_region_particles;
                below_ghost_num = num_incoming_below_ghost_particles;

            } else if (rank == n_proc - 1) { // bottom region
                // number of particles that are still in the region except the ones in ghost zone
                int remaining_particles_num = 0;
                // number of particles that are out of the region
                int send_above_particles_num = 0;
                // number of particles resides in the neighbor's region
                int send_above_ghost_particles_num = 0;

                particle_t tmp_local_particles[local_particles_num];
                particle_t send_above_buffer[local_particles_num];
                particle_t send_above_ghost_particles[local_particles_num];

                for (int i = 0; i < local_particles_num; i++) {
                    if (is_particle_in_region(local_particles[i].x, x_min, x_max)) { // particles in the region (ghost zone excluded)
                        tmp_local_particles[remaining_particles_num] = local_particles[i];
                        remaining_particles_num += 1;
                    } else { // the particles that leave the region
                        send_above_buffer[send_above_particles_num] = local_particles[i];
                        send_above_particles_num += 1;
                    }

                    if ( local_particles[i].x >= x_min && local_particles[i].x < x_min + grid_width) {
                        send_above_ghost_particles[send_above_ghost_particles_num] = local_particles[i];
                        send_above_ghost_particles_num += 1;
                    }
                }

                // collect the particles that still exist in the region
                memcpy(local_particles, tmp_local_particles, remaining_particles_num * sizeof(particle_t));

                // update the number of local particles
                local_particles_num = remaining_particles_num;

                MPI_Status receive_above_status, receive_above_ghost_status;

                particle_t receive_from_above_buffer[n - remaining_particles_num];

                // receive the new particles coming into the region
                MPI_Irecv(receive_from_above_buffer,
                          n - remaining_particles_num,
                          PARTICLE,
                          rank - 1,
                          RECEIVE_ABOVE,
                          MPI_COMM_WORLD,
                          &receive_above_request);

                // send the particles going out side the region
                MPI_Isend(send_above_buffer,
                          send_above_particles_num,
                          PARTICLE,
                          rank - 1,
                          RECEIVE_BELOW,
                          MPI_COMM_WORLD,
                          &send_above_request);

                // receive the new ghost zone particles from neighbor process
                MPI_Irecv(above_ghost_particles,
                          n,
                          PARTICLE,
                          rank - 1,
                          RECEIVE_ABOVE_GHOST,
                          MPI_COMM_WORLD,
                          &receive_above_ghost_request);

                // send the new ghost zone particles for neighbor process
                MPI_Isend(send_above_ghost_particles,
                          send_above_ghost_particles_num,
                          PARTICLE,
                          rank - 1,
                          RECEIVE_BELOW_GHOST,
                          MPI_COMM_WORLD,
                          &send_above_ghost_request);

                MPI_Wait(&receive_above_request, &receive_above_status);
                MPI_Wait(&send_above_request, MPI_STATUS_IGNORE);
                MPI_Wait(&receive_above_ghost_request, &receive_above_ghost_status);
                MPI_Wait(&send_above_ghost_request, MPI_STATUS_IGNORE);

                // get the number of received particles in region
                int num_incoming_region_particles;
                MPI_Get_count(&receive_above_status, PARTICLE, &num_incoming_region_particles);

                // get the number of received particles in ghost
                int num_incoming_above_ghost_particles;
                MPI_Get_count(&receive_above_ghost_status, PARTICLE, &num_incoming_above_ghost_particles);

                memcpy(local_particles + remaining_particles_num, receive_from_above_buffer, num_incoming_region_particles * sizeof(particle_t));
                local_particles_num += num_incoming_region_particles;
                above_ghost_num = num_incoming_above_ghost_particles;

            }
            else { // middle region
                // number of particles that are still in the region except the ones in ghost zone
                int remaining_particles_num = 0;
                // number of particles that are out of the region
                int send_below_particles_num = 0;
                int send_above_particles_num = 0;
                // number of particles resides in the neighbor's region
                int send_below_ghost_particles_num = 0;
                int send_above_ghost_particles_num = 0;

                particle_t tmp_local_particles[local_particles_num];
                particle_t send_below_buffer[local_particles_num];
                particle_t send_below_ghost_particles[local_particles_num];
                particle_t send_above_buffer[local_particles_num];
                particle_t send_above_ghost_particles[local_particles_num];

                for (int i = 0; i < local_particles_num; i++) {
                    if (is_particle_in_region(local_particles[i].x, x_min,
                                              x_max)) { // particles in the region (ghost zone excluded)
                        tmp_local_particles[remaining_particles_num] = local_particles[i];
                        remaining_particles_num += 1;
                    } 
                    else if (local_particles[i].x < x_min) { // send to above
                        send_above_buffer[send_above_particles_num] = local_particles[i];
                        send_above_particles_num += 1;
                    } 
                    // else{ // send to below
                    //     send_below_buffer[send_below_particles_num] = local_particles[i];
                    //     send_below_particles_num += 1;
                    // }


                    if (local_particles[i].x >= x_min && local_particles[i].x < x_min + grid_width) { // send to above ghost
                        send_above_ghost_particles[send_above_ghost_particles_num] = local_particles[i];
                        send_above_ghost_particles_num += 1;
                    } else if (local_particles[i].x <= x_max && local_particles[i].x >= x_max - grid_width) { // send to below ghost
                        send_below_ghost_particles[send_below_ghost_particles_num] = local_particles[i];
                        send_below_ghost_particles_num += 1;
                    }
                }

                // collect the particles that still exist in the region
                memcpy(local_particles, tmp_local_particles, remaining_particles_num * sizeof(particle_t));

                // update the number of local particles
                local_particles_num = remaining_particles_num;

                // status for getting the number of received particles
                MPI_Status receive_above_status, receive_above_ghost_status;
                MPI_Status receive_below_status, receive_below_ghost_status;

                particle_t receive_from_below_buffer[n - remaining_particles_num];
                particle_t receive_from_above_buffer[n - remaining_particles_num];

                // receive the new particles coming into the region from below
                MPI_Irecv(receive_from_below_buffer,
                          n - remaining_particles_num,
                          PARTICLE,
                          rank + 1,
                          RECEIVE_BELOW,
                          MPI_COMM_WORLD,
                          &receive_below_request);

                // receive the new particles coming into the region from above
                MPI_Irecv(receive_from_above_buffer,
                          n - remaining_particles_num,
                          PARTICLE,
                          rank - 1,
                          RECEIVE_ABOVE,
                          MPI_COMM_WORLD,
                          &receive_above_request);

                // send the particles going out side the region to below
                MPI_Isend(send_below_buffer,
                          send_below_particles_num,
                          PARTICLE,
                          rank + 1,
                          RECEIVE_ABOVE,
                          MPI_COMM_WORLD,
                          &send_below_request);

                // send the particles going out side the region to above
                MPI_Isend(send_above_buffer,
                          send_above_particles_num,
                          PARTICLE,
                          rank - 1,
                          RECEIVE_BELOW,
                          MPI_COMM_WORLD,
                          &send_above_request);

                // receive the new ghost zone particles from neighbor process
                MPI_Irecv(below_ghost_particles,
                          n,
                          PARTICLE,
                          rank + 1,
                          RECEIVE_BELOW_GHOST,
                          MPI_COMM_WORLD,
                          &receive_below_ghost_request);

                // send the new ghost zone particles for neighbor process
                MPI_Isend(send_below_ghost_particles,
                          send_below_ghost_particles_num,
                          PARTICLE,
                          rank + 1,
                          RECEIVE_ABOVE_GHOST,
                          MPI_COMM_WORLD,
                          &send_below_ghost_request);

                // receive the new ghost zone particles from above process
                MPI_Irecv(above_ghost_particles,
                          n,
                          PARTICLE,
                          rank - 1,
                          RECEIVE_ABOVE_GHOST,
                          MPI_COMM_WORLD,
                          &receive_above_ghost_request);

                // send the new ghost zone particles for neighbor above process
                MPI_Isend(send_above_ghost_particles,
                          send_above_ghost_particles_num,
                          PARTICLE,
                          rank - 1,
                          RECEIVE_BELOW_GHOST,
                          MPI_COMM_WORLD,
                          &send_above_ghost_request);

                MPI_Wait(&receive_below_request, &receive_below_status);
                MPI_Wait(&send_below_request, MPI_STATUS_IGNORE);
                MPI_Wait(&receive_below_ghost_request, &receive_below_ghost_status);
                MPI_Wait(&send_below_ghost_request, MPI_STATUS_IGNORE);
                MPI_Wait(&receive_above_request, &receive_above_status);
                MPI_Wait(&send_above_request, MPI_STATUS_IGNORE);
                MPI_Wait(&receive_above_ghost_request, &receive_above_ghost_status);
                MPI_Wait(&send_above_ghost_request, MPI_STATUS_IGNORE);


                // get the number of received particles in region from below
                int num_incoming_region_particles_from_below;
                MPI_Get_count(&receive_below_status, PARTICLE, &num_incoming_region_particles_from_below);

                // get the number of received particles in ghost from below
                int num_incoming_below_ghost_particles;
                MPI_Get_count(&receive_below_ghost_status, PARTICLE, &num_incoming_below_ghost_particles);

                // get the number of received particles in region from above
                int num_incoming_region_particles_from_above;
                MPI_Get_count(&receive_above_status, PARTICLE, &num_incoming_region_particles_from_above);

                // get the number of received particles in ghost from above
                int num_incoming_above_ghost_particles;
                MPI_Get_count(&receive_above_ghost_status, PARTICLE, &num_incoming_above_ghost_particles);

                // concatenate local particles with above
                memcpy(local_particles + remaining_particles_num, receive_from_above_buffer, num_incoming_region_particles_from_above * sizeof(particle_t));
                local_particles_num += num_incoming_region_particles_from_above;

                // concatenate local particles with below
                memcpy(local_particles + local_particles_num, receive_from_below_buffer, num_incoming_region_particles_from_below * sizeof(particle_t));
                local_particles_num += num_incoming_region_particles_from_below;

                above_ghost_num = num_incoming_above_ghost_particles;
                below_ghost_num = num_incoming_below_ghost_particles;

            }
        }

        if( find_option( argc, argv, "-no" ) == -1 )
        {

            MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);


            if (rank == 0){
                //
                // Computing statistical data
                //
                if (rnavg) {
                    absavg +=  rdavg/rnavg;
                    nabsavg++;
                }
                if (rdmin < absmin) absmin = rdmin;
            }
        }
    }
    simulation_time = read_timer( ) - simulation_time;

    if (rank == 0) {
        printf( "n = %d, simulation time = %g seconds", n, simulation_time);

        if( find_option( argc, argv, "-no" ) == -1 )
        {
            if (nabsavg) absavg /= nabsavg;
            //
            //  -the minimum distance absmin between 2 particles during the run of the simulation
            //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
            //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
            //
            //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
            //
            printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
            if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
            if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
        }
        printf("\n");

        //
        // Printing summary data
        //
        if( fsum)
            fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }

    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( all_particles );
    if( fsave )
        fclose( fsave );

    MPI_Finalize( );

    return 0;
}
