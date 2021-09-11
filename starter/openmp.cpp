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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <atomic>
#include "common.h"
#include "omp.h"

//  tuned constants
#define cutoff  0.01
#define density 0.0005


struct node {
    int index;

    node *next;
};

// Each grid is a linked_list
class Grid
{
private:
    std::atomic<node *>head = {nullptr};
public:
    Grid() = default;

    // add the node to the end of the linked_list
    void add_node(int index)
    {
        auto *tmp = new node;
        tmp->index = index;
        tmp->next = NULL;
        while (!head.compare_exchange_weak(tmp->next, tmp))
            {}
    }

    // clean the list for next timestep (clean the heap)
    void clean_list()
    {
        auto cur_node = head.load();
        while (cur_node) {
            auto pre_node = cur_node;
            cur_node = cur_node->next;
            delete pre_node;
        }
        head = nullptr;
    }

    // given node that represents the particle, compute its collision with all other particles in this cell
    void handle_particle_given_cell(node *cur_node, particle_t *particles, double *dmin, double *davg, int *navg) {
        node *tmp = head;
        while (tmp) {
            apply_force(particles[cur_node->index], particles[tmp->index], dmin,davg,navg);
            tmp = tmp->next;
        }
    }

    // compute all particles in this cell
    void compute_all_particles_in_cell(Grid *grids, particle_t *particles, int x, int y, int length, double *dmin, double *davg, int *navg) {
        node *cur_node = head;

        while (cur_node) {
            particles[cur_node->index].ax = particles[cur_node->index].ay = 0;
            // give a particle, compute it with nearby grids
            for (int i = -1; i <= 1; i++) { // iterate 3 times
                for (int j = -1; j <= 1; j++) { // iterate 3 times
                    int fixed_x = x + i;
                    int fixed_y = y + j;
                    if (fixed_x >= 0 && fixed_x < length && fixed_y >= 0 && fixed_y < length) { // valid grid
                        // given a particle and the a grid, apply force
                        grids[fixed_x * length + fixed_y].handle_particle_given_cell(cur_node, particles, dmin,davg,navg);
                    }
                }
            }
            cur_node = cur_node->next;
        }
    }
};
//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    //
    // we want to initialize a 2D matrix filled wit grids (although we 
    // implement grids in a 1d array)
    // each grid is at position i, j, accessed by matrix[i * width + j]
    // each grid contains the particles
    // O(n)
    //
    double size_t;
    size_t = sqrt( density * n );
    double grid_width = cutoff;
    // number of grids along a side
    int length = ceil(size_t / grid_width);


    // initialize the grids
    Grid grids[length * length];

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    
    #pragma omp parallel private(dmin) 
    {
        numthreads = omp_get_num_threads();
        for( int step = 0; step < 1000; step++ )
        {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            // allocate every particle to its belonging grid. O(n)
            //
            #pragma omp for
            for (int i = 0; i < n; i++) {
                // compute which grid this particle belongs to, and insert the nodes
                int row_index = particles[i].x / grid_width;
                int col_index = particles[i].y / grid_width;
                int grid_index = row_index * length + col_index;

                // add the node to its grid
                // set a lock to each grid so that add_node is thread-safe 
                grids[grid_index].add_node(i);
            }

            #pragma omp for reduction (+:navg) reduction(+:davg)
            for (int x = 0; x < length; x++) {
                for (int y = 0; y < length; y++) {

                    grids[x * length + y].compute_all_particles_in_cell(grids, particles, x, y, length, &dmin,&davg,&navg);
                }
            }

            //
            // free the nodes in every grid
            //
            #pragma omp for
            for (int i = 0; i < length * length; i++) {
                grids[i].clean_list();
            }

            //
            //  move particles
            //
            #pragma omp for
            for( int i = 0; i < n; i++ ) 
                move( particles[i] );
    
            if( find_option( argc, argv, "-no" ) == -1 ) 
            {
            //
            //  compute statistical data
            //
            #pragma omp master
            if (navg) { 
                absavg += davg/navg;
                nabsavg++;
            }

            #pragma omp critical
        if (dmin < absmin) absmin = dmin; 
            
            //
            //  save if necessary
            //
            #pragma omp master
            if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, particles );
            }
        }
    }

    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

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
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
