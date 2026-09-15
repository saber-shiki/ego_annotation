# Astra connection-establishment repair

Scope: patch of the fixed `gpt-6-astra` transport, not a new pipeline version or
annotation-feasibility claim. The observed blocker is one DNS answer timing out
before TLS while another address for the same hostname is reachable.

Before any HTTP request bytes, resolve the unchanged HTTPS endpoint once and try
unique IPv4/IPv6 stream addresses in resolver order. Each address has a shared
5-second TCP-plus-TLS budget; all connection work has a 20-second monotonic budget.
Elapsed DNS and CA-context setup time count against that budget, but the standard
blocking `getaddrinfo` resolver cannot itself be interrupted by this policy: DNS
can exceed 20 seconds, after which no connection is attempted. No resolver threads,
DNS override, hardcoded IP, proxy, alternate model, or alternate endpoint are used.

Use default CA verification, mandatory hostname verification, and original-host
SNI and HTTP Host. A successfully verified socket is transferred to a single HTTP
connection with automatic reconnect disabled. Send the original audited POST bytes
once, retain the 120-second socket read timeout, and never retry after send begins
(including send/read/status/JSON/schema/model failures). Redirects are rejected,
not followed. Close failed sockets, successful sockets, and responses explicitly.

The fixed policy is included in `astra_stage_inputs` and request provenance so
planned stage identity changes with the repair. When `audit_dir` is supplied,
`connection_attempts.json` binds credential-free DNS/address/phase/timing outcomes
to the exact payload hash; it never records headers, exception messages, or keys.
Existing exact-payload and response identity/schema checks remain authoritative.

CPU tests must discriminate pre-send bad-address recovery from forbidden POST
replay, exercise deadlines, certificate failure and cleanup, preserve image bytes
and response validation, and verify credential-free auditing. No authenticated
POST or inference is part of this patch. Parent integration still requires review,
same-host unauthenticated TLS/HEAD preflight, and the separately authorized fresh
pipeline run; CPU success is not evidence of annotation completion.
