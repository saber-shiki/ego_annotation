# Bounded full-timeline rigidity review transport

## Failure

Fresh0618e91milk run reached rigidity_evidence but its single rigidity POST
failed with TLS write timeout. It had15ten-frame sheets covering150frames;
re-encoding its inputs offline gives9,599,390request bytes. Unlike mask and
trajectory reviews, rigidity had neither byte budget nor exact wire audit.
Large upload is a concrete exposure, not a proven gateway-size-limit diagnosis;
TLS write failure leaves delivery uncertain, so the frozen request is not
silently replayed.

## Intervention

Use the existing finite full-sheet review encoder for rigidity, with a new
explicit3,000,000-byte limit (previous rigidity budget was unbounded). Preserve
all15sheets,150frame facts, sheet order and component-motion judgment in a
single request. Do not split/reconcile separate rigidity answers or promote
unresolved/articulated judgments. Keep the existing5-tier ladder ending at
JPEG65/75% dimensions and the original120s read/write socket setting. Other
judgment budgets are unchanged. The frozen fixture fits at2,693,295bytes;
3MB allows this fidelity floor without more aggressive scaling or dropped
frames. If the ladder cannot fit another input, retain its exact oversized
payload and fail before HTTP.

Bind source sheet paths/hashes and ordered frame mappings from motion_sequences
into typed review_images. Include the resulting evidence and encoder policy in
planned stage contracts. Require an audit directory for live rigidity judgments,
including both public CLI forms. Wire bytes, tier sizes, original/effective image
hashes, connection attempts and model identity are saved. Original motion sheets
and physical observations are untouched; compressed review pixels do not replace
physical measurements. Subtle motion can remain uncertain at lower fidelity.

## Validation

CPU tests cover input immutability, exact sequence/path/hash/frame mapping,
budget/audit forwarding, original nonrigid-answer preservation, CLI audit path,
and cold ask planned/actual request consistency. Existing generic encoder tests
cover tier exhaustion and no dropping. Run one explicitly diagnostic Astra
request using frozen150-frame evidence, but write only outside that run; preserve
any answer or failure rather than coercing rigid. New production code requires a
new immutable snapshot and fresh raw run, never cross-code resume or restamping.
