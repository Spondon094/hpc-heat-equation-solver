#include "Solver.h"
#include "Grid.h"
#include <omp.h>

// constructor initializer list: specifies initial values for one or more data members of the object being created
// this constructor is defined outside the class body
// CLASS:: constructor's_name(parameters):members_initialized{}


SolverClass::SolverClass(PDE *pde_, Grid *x_, Grid *b_):pde(pde_),x(x_),b(b_)
{
}

// Defining CG function outside the class belonging to class SolverClass

int SolverClass::CG(int niter, double tol)
{
    Grid *p = new Grid(pde->numGrids_x(), pde->numGrids_y()); // p points to a dynamically allocated initialized object of type Grid
    Grid *v = new Grid(pde->numGrids_x(), pde->numGrids_y()); // do p and v point to the same memory address

    int iter = 0;
    double lambda = 0;
    double alpha_0 = 0, alpha_1 = 0;
    //Calculate residual
    //p=A*x
    pde->applyStencil(p,x);
    //p=b-A*x
    axpby(p,1,b,-1,p);
    //calculate alpha_0
    alpha_0 = dotProduct(p,p);

    Grid *r = new Grid(*p);

    START_TIMER(CG);

    while( (iter<niter) && (alpha_0>tol*tol) )
    {
        pde->applyStencil(v,p);
        lambda =  alpha_0/dotProduct(v,p);
        //Update x
        axpby(x, 1.0, x, lambda, p);
        //Update r
        axpby(r, 1.0, r, -lambda, v);
        alpha_1 = dotProduct(r,r);
        //Update p
        axpby(p, 1.0, r, alpha_1/alpha_0, p);
        alpha_0 = alpha_1;
#if DEBUG
        printf("iter = %d, res = %.15e\n", iter, alpha_0);
#endif
        ++iter;
    }

    STOP_TIMER(CG);

    delete p;
    delete v;
    delete r;

    return (iter);
}

int SolverClass::PCG(int niter, double tol)
{
    Grid* r = new Grid(pde->numGrids_x(), pde->numGrids_y());
    Grid* z = new Grid(pde->numGrids_x(), pde->numGrids_y());
    Grid* v = new Grid(pde->numGrids_x(), pde->numGrids_y());

    int iter = 0;
    double lambda = 0;
    double alpha_0 = 0, alpha_1 = 0;
    double res_norm_sq = 0;
    //Calculate residual
    pde->applyStencil(r,x);
    axpby(r,1,b,-1,r);
    res_norm_sq = dotProduct(r,r);
    pde->GSPreCon(r,z);

    alpha_0 = dotProduct(r,z);
    Grid* p = new Grid(*z);

    START_TIMER(PCG);

    while( (iter<niter) && (res_norm_sq>tol*tol) )
    {
        pde->applyStencil(v,p);
        lambda =  alpha_0/dotProduct(v,p);
        //Update x
        axpby(x, 1.0, x, lambda, p);
        //Update r
        axpby(r, 1.0, r, -lambda, v);
        res_norm_sq = dotProduct(r,r);
        //Update z
        pde->GSPreCon(r, z);
        alpha_1 = dotProduct(r,z);
        //Update p
        axpby(p, 1.0, z, alpha_1/alpha_0, p);
        alpha_0 = alpha_1;

#if DEBUG
        printf("iter = %d, res = %.15e\n", iter, res_norm_sq);
#endif
        ++iter;
    }

    STOP_TIMER(PCG);

    delete r;
    delete z;
    delete v;
    delete p;

    return (iter);
}
