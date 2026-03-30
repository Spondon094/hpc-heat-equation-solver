#ifndef GRID_H
#define GRID_H

#include<iostream>
#include<vector>
#include<assert.h>
#include <iomanip>
#include<fstream>
#include<sstream>
#include<vector>
#include<math.h>
#include <functional>
#include "types.h"
#include "timer.h"

#ifndef HALO
    #define HALO 1
#endif

struct Grid{
    int columns, rows;
    double* arrayPtr;
    BC_TYPE ghost[4];

    /* @brief Class constructor which takes columns and rows
     *
     * @param columns number of columns in inner domain (boundary will be
     *                  allocated automatically)
     * @param rows number of rows in inner domain (boundary will be allocated
     *                  automatically)
     */
    Grid(int columns,int rows);
    /* @brief Class constructor which takes columns and rows , and boundary to initialise halo
     * the rows and columns passed is that of actual domain (excluding halo)
     *
     * @param columns number of columns in inner domain (excl. boundary)
     * @param rows number of rows in inner domain (excl. boundary)
     * @param ghost Array of Boundary type on NORTH,EAST,WEST,SOUTH; POSSIBLE value for elements are Neumann and Dirichlet
     */
    Grid(int columns,int rows, BC_TYPE *ghost);
    //copy constructor; Performs deep copy
    Grid(const Grid& s);

    /*@brief Element access operators
     * @param row number
     * @param column number
     */
    inline double& operator()(int,int);
    inline double& operator()(int,int) const;

    //print the Grid
    void print(bool halo=false);

    //read the grid values from a file
    bool readFile(const std::string& name, bool halo=false);

    //write the grid values to a file
    bool writeFile(const std::string& name, bool halo=false);

    /* @brief returns number of grids in x direction
     * @param halo Whether halo should be counted
     */
    int numGrids_x(bool halo=false) const;
    /* @brief returns number of grids in y direction
     * @param halo Whether halo should be counted
     */
    int numGrids_y(bool halo=false) const;
    /* @brief returns total number of grids in x and y direction
     * @param halo Whether halo should be counted
     */
    int numGrids(bool halo=false) const;
    /* @brief fills the whole matrix using the passed in function evaluated at (i,j)
     *
     * @param func The function with which matrix has to be set
     */
    void fill(std::function<double(int,int)> func, bool halo=false);
    void fill(double val, bool halo=false);
    void rand(bool halo=false, unsigned int seed=1);
    /* @brief fills the boundary points on corresponding Direction(param2) with the function passed(param1)
     *
     * @param func The function with which boundary is to be set
     * @param dir  Which boundary is to be set. Possible values NORTH,EAST,WEST,SOUTH
     */
    void fillBoundary(std::function<double(int,int)> func, Direction dir);
    /* @brief copies the inner points to halo on corresponding Direction(param2) with a shift as passed by function(param1)
     *
     * @param func The function with which boundary is to be set
     * @param dir  Which boundary is to be set. Possible values NORTH,EAST,WEST,SOUTH
     */
    void copyToHalo(std::function<double(int,int)> func, Direction dir);
    /* @brief swaps 2 matrices
     *
     * @param the other matrix with which current one(*this) has to be swapped
     */
    void swap(Grid &other);
    /* Class Destructor
     *
     */
    ~Grid();

};



inline double& Grid::operator ()(int i, int j ){
#ifdef DEBUG
    assert( i >= (0) && j >= (0) );
    assert( i< (rows) && j < (columns) );
#endif
    return  arrayPtr[(i)*columns+(j)];
}

inline double& Grid::operator ()(int i, int j ) const {
#ifdef DEBUG
    assert( i >= (0) && j >= (0) );
    assert( i< (rows) && j < (columns) );
#endif
    return  arrayPtr[(i)*columns+(j)];
}

bool writeGnuplotFile(const std::string& name, Grid &src, double lx, double ly, bool halo=false);

void axpby(Grid *res, double a, Grid *x, double b, Grid *y, bool halo=false);
void copy(Grid *res, double a, Grid *x, bool halo=false);
double dotProduct(Grid *x,Grid *y, bool halo=false);
bool isSymmetric(Grid *u, double tol=1e-10, bool halo=false);

#endif // GRID_H
