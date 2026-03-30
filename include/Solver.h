#ifndef SOLVER_H
#define SOLVER_H
#include "PDE.h"

//forward declaration
class PDE;

class SolverClass
{
    PDE *pde;
    Grid *x;
    Grid *b;
    public:
    //constructor
    //Solves for x: Ax = b: A defined by operator in PDE
    SolverClass(PDE *pde_, Grid *x_, Grid *b_);

    //cg
    int CG(int niter, double tol);
    //pcg
    int PCG(int niter, double tol);
};

#endif
