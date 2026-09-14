"""Sole numerical dispatch; results remain candidates, never physical acceptance."""
from optimization.problems import StandaloneScene, PreparedVisualFit
from optimization.object.pose import solve_scene_pose

def solve(problem):
    if isinstance(problem, StandaloneScene):
        return solve_scene_pose(problem.nodes, problem.times, problem.edges, problem.anchor,
                                problem.config, problem.reprojections)
    if isinstance(problem, PreparedVisualFit):
        from optimization.interaction.joint_fit import fit_visual
        return fit_visual(problem)
    raise TypeError(f"unreviewed numerical problem: {type(problem).__name__}")
