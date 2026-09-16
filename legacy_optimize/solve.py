"""Sole numerical dispatch; results remain candidates, never physical acceptance."""
from legacy_optimize.problems import StandaloneScene, PreparedVisualFit, FormalManoInterval, ObservedObjectPose, StandaloneShape, FormalObjectTrajectory
from legacy_optimize.object.pose import solve_scene_pose

def solve(problem):
    if isinstance(problem, StandaloneScene):
        return solve_scene_pose(problem.nodes, problem.times, problem.edges, problem.anchor,
                                problem.config, problem.reprojections)
    if isinstance(problem, PreparedVisualFit):
        from legacy_optimize.interaction.joint_fit import fit_visual
        return fit_visual(problem)
    if isinstance(problem, FormalManoInterval):
        from legacy_optimize.hand.fit import fit_formal_interval
        return fit_formal_interval(problem.arguments())
    if isinstance(problem, ObservedObjectPose):
        from observations.object.pose import fit_observed_pose
        return fit_observed_pose(list(problem.arguments))
    if isinstance(problem, StandaloneShape):
        from legacy_optimize.object.shape import bind
        return bind(problem.posterior,problem.annotations,problem.appearance,problem.decision,
                    problem.native_qc,problem.output,problem.object_id,problem.run_root)
    if isinstance(problem, FormalObjectTrajectory):
        from legacy_optimize.object.pose import fit_formal_object
        return fit_formal_object(problem.arguments())
    raise TypeError(f"unreviewed numerical problem: {type(problem).__name__}")
