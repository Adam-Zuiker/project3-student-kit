# Project 3 (Forward Kinematics) — student kit

Everything you need for Project 3, without the rest of the course repo.

- **`docs/PROJECT3_*.md`**: start here. This is the assignment handout: overview, learning
  goals, the checkpoint ("Zero Configuration FK Transforms"), the final requirements, and how
  you are graded.
- **`spec/FK_API.md`**: the exact contract. It covers the node graph (`param_server`,
  `robot_state_publisher`, `robot_world_state_publisher`, and the recommended
  `joint_state_publisher` and `finite_state_machine`), the `/param_server/set_param` and
  `/param_server/get_param` services, `robot_description` and its description status,
  `/joint_states` in, `/tf` and `/xform_world` out, and what `make run` and `make demo` must
  start.
- **`spec/ROBOT_DESCRIPTION.md`**: which URDF descriptions are valid, which elements and
  attributes are read, and the defaults that apply when they are missing.
- **`spec/ROSBRIDGE_PROTOCOL.md`**: the external TCP/JSON transport on `127.0.0.1:9095`, reused
  unchanged from Projects 1 and 2. Project 3 requires your gateway to accept request lines of
  at least 1 MiB, because real robot descriptions are large (see `spec/FK_API.md`).
- **`starter/`**: four alternative starting points (`c/`, `cpp/`, `python/`, `rust/`). Choose
  **one**. Each is a placeholder: a `Makefile` (`build`, `run`, `demo`, `clean`), a `main`
  program that prints a `TODO` line, and a small example of reading a URDF with that language's
  bundled XML parser. The starters include no middleware and no nodes: bring your Project 1
  middleware and write the nodes yourself. See `starter/README.md`.
- **`robots/fetch/fetch.urdf`**: a real robot description (the Fetch mobile manipulator) to try
  your runtime on, redistributed unmodified under CC BY-NC-SA 4.0. See
  `robots/fetch/LICENSE.md`.
- **`make_submission.sh`**: once your chosen starter directory builds and runs, run
  `./make_submission.sh <c|cpp|python|rust>` to produce a correctly shaped
  `submission.tar.gz`. Its `Makefile` is at the archive root, which is what the autograder
  expects.
