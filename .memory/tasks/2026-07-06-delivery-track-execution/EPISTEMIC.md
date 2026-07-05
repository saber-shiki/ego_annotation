# Current epistemic model

The delivery task is a productization problem, not a HOI research problem. The required artifact is a directly callable annotation API producing numeric head/camera, hand, semantic clip, validation, provenance, error, and throughput outputs. Renders are QC/demo views of those numbers.

The 5mm ideal remains the optimization target across head/camera, wrist/root, all-joint MPJPE, hand surface/MPVPE, projection, visibility, and temporal stability. Current evidence says each axis may have a different measured frontier; the work is to measure the frontier and choose the next error-reduction mechanism per axis.

Head/camera accuracy is an estimator-and-measurement program. The API reserves optional device calibration/VIO/head-pose fields but cannot assume they are present. The selected promotion benchmark is an in-house fixed-gauge fiducial/mocap lockbox; public sidecar datasets are development/regression checks.

Hand accuracy uses wrist/root as near-term anchor and all-joint MPJPE as first post-root optimization priority. MPVPE/surface, visibility, projection, and jitter remain protected metrics.

Serving should begin with FastAPI job ingress plus Ray Serve/Ray GPU actors because arbitrary PyTorch functions, persistent residency, batching, and video-affinity scheduling matter more than a pure tensor-serving ABI at this stage.

The strict next uncertainty is M0/M1: define the exact metric vector and evaluator artifacts, then measure current head/camera and hand frontiers on frozen development clips before changing model stages.
