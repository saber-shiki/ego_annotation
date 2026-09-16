"""Sole numerical dispatch; results remain candidates, never physical acceptance."""
from optimization.problems import StandaloneScene, PreparedVisualFit, ObjectSurfaceEnvelope, FormalManoInterval, ObservedObjectPose, StandaloneShape, FormalObjectTrajectory
from optimization.object.pose import solve_scene_pose

def solve(problem):
    if isinstance(problem, StandaloneScene):
        return solve_scene_pose(problem.nodes, problem.times, problem.edges, problem.anchor,
                                problem.config, problem.reprojections)
    if isinstance(problem, ObjectSurfaceEnvelope):
        from optimization.object.surface_fit import SurfaceProblem, fit_surface
        from gates.object.surface_candidate import validate_surface_candidate
        candidate = fit_surface(SurfaceProblem(
            annotations=problem.annotations, refined_posterior=problem.refined_posterior,
            mesh_binding=problem.mesh_binding, envelope_report=problem.envelope_report,
            appearance=problem.appearance, run_root=problem.run_root,
            object_id=problem.object_id, output_dir=problem.output,
            profile=problem.profile))
        validate_surface_candidate(candidate, object_id=problem.object_id, run_root=problem.run_root)
        return candidate
    if isinstance(problem, PreparedVisualFit):
        from optimization.interaction.joint_fit import fit_visual
        return fit_visual(problem)
    if isinstance(problem, FormalManoInterval):
        from optimization.hand.fit import fit_formal_interval
        return fit_formal_interval(problem.arguments())
    if isinstance(problem, ObservedObjectPose):
        from optimization.object.pose import fit_observed_pose
        return fit_observed_pose(list(problem.arguments))
    if isinstance(problem, StandaloneShape):
        from optimization.object.shape import bind
        return bind(problem.posterior,problem.annotations,problem.appearance,problem.decision,
                    problem.native_qc,problem.output,problem.object_id,problem.run_root,
                    object_profile=problem.object_profile, geometry_profile=problem.geometry_profile,
                    pointmap_receipt=problem.pointmap_receipt)
    if isinstance(problem, FormalObjectTrajectory):
        from optimization.object.pose import fit_formal_object
        return fit_formal_object(problem.arguments())
    raise TypeError(f"unreviewed numerical problem: {type(problem).__name__}")
