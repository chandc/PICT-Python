// Re_tau 180 minimal channel, isotropic graded triangles: h = max(h_wall, slope*d_wall) capped at h_max.
// Periodic in x (matched nodes on left/right), no-slip walls bottom/top. Physical tags 1..4 as rect_mesh.
Lx = Pi; Ly = 2.0;
hw = 2.0/180.0;      // wall cell size, 2 wall units
hmax = 16.0/180.0;   // core cell size, 16 wall units
slope = 0.3;
Point(1) = {0, 0, 0, hw}; Point(2) = {Lx, 0, 0, hw}; Point(3) = {Lx, Ly, 0, hw}; Point(4) = {0, Ly, 0, hw};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4}; Plane Surface(1) = {1};
Periodic Curve {2} = {4} Translate {Lx, 0, 0};
Field[1] = MathEval; Field[1].F = Sprintf("Max(%g, Min(%g, %g*Min(y, %g-y)))", hw, hmax, slope, Ly);
Background Field = 1;
Mesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6; Mesh.Smoothing = 10;
Physical Curve("left", 1) = {4}; Physical Curve("right", 2) = {2}; Physical Curve("bottom", 3) = {1}; Physical Curve("top", 4) = {3};
Physical Surface("fluid", 5) = {1};
