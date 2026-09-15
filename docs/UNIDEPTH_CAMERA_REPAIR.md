# UniDepth camera/image preprocessing repair

Patch scope: repair the deterministic RGB/K mismatch in the deployed UniDepthV2
`infer` path; no provider, weight, threshold or shared-source changes.

## Cause and intervention

The deployed `get_resize_factor` computes nominal scale `s` from pixel bounds,
then rounds each image dimension up to a multiple of 14. `infer` resizes RGB to
those rounded dimensions but scales its mutable camera by `s`. Its rays use
half-integer centers; project camera contracts use integer centers. This makes
the supplied rays wrong **before the depth head**, not just in the export.

The local adapter retains the deployed inference, normalization, autocast,
padding, interpolation, depth head and confidence/radius extraction. For original
shape `(H,W)`, padding `(l,r,t,b)`, padded size `(Hp,Wp)`, rounded size `(Hr,Wr)`,
let `ax=Wr/Wp`, `ay=Hr/Hp`. Integer input K is converted to padded upstream K by
adding `(l+0.5,t+0.5)` to its principal point. Desired network K has
`fx'=ax*fx`, `fy'=ay*fy`, `cx'=ax*(cx+l+0.5)`,
`cy'=ay*(cy+t+0.5)`. To make deployed `infer` produce precisely this K, pass
`fx_in=fx'/s`, `fy_in=fy'/s`, `cx_in=cx'/s-l`, `cy_in=cy'/s-t`.
The passed matrix is private storage; caller K is never shared with the provider.

The camera head is independent of supplied conditioning. Its returned K has
already been nominally unscaled and unpadded by upstream. Correct **that head K**
using `fx_local=fx_returned*s/ax`,
`cx_local=(cx_returned+l)*s/ax-l-0.5`, and the analogous y formulas.
Never substitute supplied K for this diagnostic. With no input K the same
inverse transform is required, and the head K remains the inferred active camera.

Returned rays are untouched (upstream interpolates unit rays then renormalizes).
Small interpolation/clamping residuals, especially at edges, remain measurable;
no exact-ray replacement hides them. Upstream radius remains the norm of its
postprocessed points and confidence remains its postprocessed confidence. At the
final export plane radius is resized then projected to camera z using the exact
contract K, or the correctly center-scaled inferred K when unconditioned.

## Supported interface and checks

This is an audited adapter for deployed UniDepthV2 `infer(rgb,camera=...)` plus
`encode_decode(inputs,image_metas=...)`, bilinear interpolation, shape multiple
14, and the deployed shape-bound/resolution-level algorithm. Unsupported
interfaces/configurations and invalid pinhole matrices fail explicitly; no
fallback provider or uncorrected inference path. Re-audit on provider updates.
CPU tests use a fake provider reproducing deployed mutable K preprocessing,
half-center ray generation at the encoder/decoder seam and postprocessing. They
check the actual wrapper call, image/camera alignment, predicted-vs-supplied K,
repeatability, confidence/radius and the old 2.049125-degree regression.

## Discriminating validation still required

No model inference is run in this patch. Parent's fresh offline A800 milk-frame
check must capture the actual depth-head conditioning rays/K, returned ray error,
and radius/depth changes against the old failure. Correct network K and reduced
ray error establish the geometric repair; inert radius requires checking depth
head coupling, not output relabeling. Persist any failure and inspect depth
quality visually. Review and full grouped/supplementary/HOT3D regression precede
integration. This is not full-video or annotation feasibility evidence.
