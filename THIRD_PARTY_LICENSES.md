# Third-Party Go Module Licenses

Modules compiled into the root `aikit` module (this `go.mod` / `go.sum`).
Test-only modules (reachable only via `*_test.go`) are excluded. The gpu/\*
backends and `chunk/treesitter` are separate modules with their own
`go.mod`/`go.sum` (cgo dependencies in particular — `gotreesitter` for
`chunk/treesitter`, CUDA/Metal bindings for `gpu/*`) and are not covered
by this list.

Regenerate with `go-licenses report ./...` (see
[github.com/google/go-licenses](https://github.com/google/go-licenses); v2,
matching the Go toolchain in `go.mod` — an older major version misreads
modern stdlib module metadata and misreports every stdlib package as
missing module info) after `go mod tidy`. The standard library is governed
by Go's own [BSD-3-Clause license](https://go.dev/LICENSE) and is not
re-listed here.

Generated 2026-08-15 from `go-licenses report ./...` against `go.sum`.

For the bundled `potion-code-16M` model weights (MIT) and their upstream
attribution chain (Apache-2.0 for `snowflake-arctic-embed-m-long`), see
[`NOTICE`](NOTICE).

| Module | Version | License |
|---|---|---|
| `golang.org/x/sys` | `v0.47.0` | BSD-3-Clause |
| `golang.org/x/text` | `v0.40.0` | BSD-3-Clause |

Both licenses above are permissive and redistribution-compatible. Each
module's upstream `LICENSE` file remains the authoritative grant.
