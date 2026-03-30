#ifndef PDE_H
#define PDE_H

#include "Grid.h"
#include "types.h"
#include "Solver.h"
#include <functional>

class PDE{

    public:
        //constructor
        PDE(int len_x_, int len_y_, int grids_x_, int grids_y_);
        /* @brief number of grids in x direction
         * @param set true if halo has to be also included
         */
        int numGrids_x(bool halo=false);
        /* @brief number of grids in y direction
         * @param set true if halo has to be also included
         */
        int numGrids_y(bool halo=false);
        /*@brief Initialises grids
         *
         */
        void init(Grid *grid);
        /*@brief apply Dirichlet boundary conditions on Dirichlet boundaries
         * as specified by 'boundary' with 'boundary_func'
         * @param u Grid on which boundary is to be set
         */
        void applyBoundary(Grid *u);
        /*@brief Refreshes Neumann boundaries
         */
        void refreshBoundary(Grid *u);
        //rhsFunc : function with which rhs is initialized
        double (*initFunc) (int x, int y, double h_x, double h_y);
        BC_TYPE boundary[4];
        //boundary_func: An array of function to be applied at NORTH, EAST, WEST, SOUTH boundaries
        double (*boundaryFunc[4]) (int x, int y, double h_x, double h_y); //array for boundary functions
        /*@brief Applies PDE operator on u i.e. res=A*u
         *
         * @param res The Grid where result has to be stores
         */
        void applyStencil(Grid *res, Grid *u);
        /*@brief Applies one complete forward and backward substitution
         * required for GS preconditioning
         *
         * @param rhs: RHS grid
         * @param u: Solution(unknown) grid
         * @param scale_rhs: scale rhs with diag
         * @param rev: In rev direction
         */
        void GSPreCon(Grid *rhs, Grid *u);
        /*@brief Solves for x:A*x=b
         *
         * @param x: Unknown Grid
         * @param b: RHS Grid
         * @param type: type of solver to be used options CG or  PCG
         * @param tol: tolerance to be achieved optional
         */
        int solve(Grid *x, Grid *b, Solver type, int niter, double tol=1e-8);

    private:
        int len_x, len_y;
        int grids_x, grids_y;
        double h_x, h_y;
};

#endif // PDE_H
