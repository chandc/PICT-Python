lc = 1.0/16;
Point(1)={0,0,0,lc}; Point(2)={1,0,0,2.0*lc}; Point(3)={1,1,0,lc}; Point(4)={0,1,0,2.0*lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Curve Loop(1)={1,2,3,4}; Plane Surface(1)={1};
Physical Curve("bottom",1)={1}; Physical Curve("right",2)={2}; Physical Curve("top",3)={3}; Physical Curve("left",4)={4};
Physical Surface("fluid",5)={1};
Mesh.Algorithm=8; Mesh.RecombinationAlgorithm=1; Mesh.RecombineAll=1; Recombine Surface{1};
