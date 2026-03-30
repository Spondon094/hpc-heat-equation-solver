#ifndef TYPE_H
#define TYPE_H

enum BC_TYPE{Dirichlet, Neumann};
enum Direction{NORTH=0,SOUTH=1,EAST=2,WEST=3,CENTER=4};
enum Solver{CG=0, PCG=1};

#define UNUSED(x) (void)(x)

#endif
