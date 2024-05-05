// generic-ish pentagon 2D domain geometry
// usage:  gmsh -2 pentagon.geo

cl = 0.2;  // characteristic length
smcl = 0.05;  // smaller length
Point(1) = {2.0,0.0,0,cl};
Point(2) = {1.0,1.0,0,cl};
Point(3) = {-1.0,1.0,0,smcl};
Point(4) = {-2.0,0.0,0,cl};
Point(5) = {0.7,-1.5,0,smcl};
Line(6) = {1,2};
Line(7) = {2,3};
Line(8) = {3,4};
Line(9) = {4,5};
Line(10) = {5,1};

Line Loop(21) = {6,7,8,9,10};
Plane Surface(22) = {21};

Physical Line(31) = {6,7,8,9,10};  // boundary will be Neumann (index = 31)

Physical Surface(41) = {22};

