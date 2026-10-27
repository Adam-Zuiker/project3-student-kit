# Boston Dynamics Atlas (v5) — URDF and meshes

Downloaded unmodified, the same way https://github.com/tkoolen/AtlasRobot.jl
does in its `deps/build.jl`, from Drake at a pinned commit:

    https://raw.githubusercontent.com/RobotLocomotion/drake/73a8da32cd41ff7fd023c3680f8250860cbd0e6b/examples/atlas/
      urdf/atlas_convex_hull.urdf  ->  urdf/atlas.urdf
      urdf/meshes/*.obj            ->  urdf/meshes/  (the 45 files AtlasRobot.jl lists)

License: Drake's BSD 3-Clause License, copied to `LICENSE.txt`.
Credit when shown: "Atlas robot model from the Drake project (RobotLocomotion), BSD 3-Clause; via AtlasRobot.jl".

Meshes are referenced as `package://Atlas/urdf/meshes/...`; the viewer bridge maps
package `Atlas` to this directory. The root link is `pelvis`; standing on flat
ground at zero configuration, the pelvis is ~0.938 m up (foot frames 0.862 m
below it, soles a further 0.076 m).

Used only by the local viewer and demo; never part of the graded submission.
