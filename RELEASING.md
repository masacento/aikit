# Releasing

This repo ships **two independently-versioned Go modules**, and they have different
release rituals. Read the one you're tagging.

## Root module — `github.com/townsendmerino/aikit` (tags `vX.Y.Z`)

Automated and CI-enforced.

1. Add a `## [X.Y.Z] — <date>` section to `CHANGELOG.md` and its `[X.Y.Z]:` compare
   link at the bottom. Additive changes bump the **minor**; bug fixes the **patch**.
2. Run the gate locally first: `scripts/release-gate.sh X.Y.Z` — it checks the CHANGELOG
   section, `golangci-lint`, `apidiff` (no Hard-tier API breakage vs the previous tag),
   and the cgo-free dependency invariant.
3. Run `scripts/vulncheck.sh` and **put its `STATEMENT:` line in the release notes.**
   This is a deliverable, not hygiene: aikit's pitch is a static binary someone scps
   somewhere and runs offline, and a binary deployed that way cannot be patched in
   place by whoever is running it. "What is in it, and is any of it known-vulnerable"
   is part of the same claim as cgo-free and dependency-light, both of which already
   have gates. `govulncheck` is reachability-filtered, so a clean result is a statement
   about the binary rather than about the dependency list.
4. Push the prep commit, wait for **root CI green**, then push the tag `vX.Y.Z`.
   `.github/workflows/release.yml` re-runs the gate on the tag and publishes the GitHub
   Release from the CHANGELOG section.

## GPU submodule — `github.com/townsendmerino/aikit/gpu` (tags `gpu/vX.Y.Z`)

**Root CI does NOT exercise this module.** It is a separate module (`gpu/go.mod`) and its
device tests need a real GPU, so they are hand-run — nothing in `go test ./...` at the
root touches `gpu/`, and no CI job runs it. A future edit to `gpu/metal.go` (or the CUDA
side) that removes or inverts a guard — for example the `setPreciseMath` fast-math
read-back that keeps a "precise" library from silently being fast-math — will **not** turn
anything red. The release ritual is the enforcement, so do not skip it:

> **Before cutting a `gpu/vX.Y.Z` tag, run the `gpu/` test suite by hand on a Mac with a
> GPU, and record the machine and macOS version in the tag message.** Root CI does not
> exercise this module.

Concretely:

0. **Before any of this: `scripts/preflight.sh`.** It runs what CI's root-module job
   runs — gofmt, build, vet, golangci-lint, cgo-free build, tests — in about a minute,
   and treats a MISSING linter as a failure rather than a skip. Install it as a hook so
   it is not a thing to remember: `ln -sf ../../scripts/preflight.sh .git/hooks/pre-push`.

   > Added 2026-08-12, after using CI as a first linter rather than a last check cost
   > three round trips in one evening — one of them a genuine `unused` finding that the
   > already-installed local linter reports in three seconds. Two of the other failures
   > that evening were the lint action failing to download its own binary from GitHub's
   > CDN (a 503, then a socket hang up), which is why CI now installs it through the Go
   > module proxy (`install-mode: goinstall`) instead.

0b. **Run the mechanical gate: `go run -C tools ./gpugate`.** Paste its `VERDICT:` line
   into the tag message. It needs NVRTC but **no GPU**, so it runs on any box; if the PTX
   group reports NVRTC unfindable, `pip install nvidia-cuda-nvrtc-cu12==12.9.86
   nvidia-cuda-runtime-cu12` or point `NVRTC_LIB` at an existing install. On darwin that
   group is n/a — NVIDIA ships no NVRTC for Apple Silicon — so a Mac's verdict covers the
   rest and says so.

   The `gpu-kernels` CI job runs the same kernel assertions on every push; the gate is
   still the enforcement point (CI is advisory here by choice — see the note at the end).
1. **Run the device half: `go run -C tools ./gpudevice`, on BOTH a Mac and an NVIDIA
   box.** Paste both `VERDICT:` lines into the tag message — this is the half the
   mechanical gate cannot do for you. It runs all nine gpu modules' suites and reports
   each ok / n/a / FAIL; the `n/a` state (the four `*metal` on Linux, the four `*cuda` on
   darwin) is why **no single machine can cover all nine** — a Mac's verdict and an NVIDIA
   box's are different halves, and a `gpu/vX.Y.Z` tag needs both.

   > It replaced `cd gpu && CGO_ENABLED=0 go test ./...`, whose `./...` stopped at module
   > boundaries and so tested one of the nine modules and silently skipped the other eight
   > (found 2026-08-12). The eight it missed are the same eight that had never been tagged
   > (below): both are what you get when a tree of sibling modules is treated as one.
2. Push as `townsendmerino` (a second account on this machine 403s on `townsendmerino`
   repos), and author commits as `townsendmerino@gmail.com`.

   > A bare `gh auth switch --user townsendmerino` is a **no-op whenever `GITHUB_TOKEN`
   > is set** — gh resolves that variable before `hosts.yml` and refuses to switch,
   > printing "The value of the GITHUB_TOKEN environment variable is being used for
   > authentication". Clear it for the switch only:
   > `env -u GITHUB_TOKEN -u GH_TOKEN gh auth switch --user townsendmerino`.
   > Note also that git reaches GitHub through `gh auth git-credential` (see
   > `~/.gitconfig`), so the identity is whatever gh resolves — `credential.helper
   > store` is not consulted for github.com, and the `x-access-token` username git
   > reports is how gh presents a token, not a second account.
3. Tag **annotated**, with the environment in the message, so the "tested on" record
   travels with the tag:
   ```
   git tag -a gpu/vX.Y.Z -m "gpu vX.Y.Z — <what changed>
   tested: <machine>, macOS <version>, gpu/ suite green"
   git push origin gpu/vX.Y.Z
   ```
   Bump the **patch** for a fix/hardening with no API change, the **minor** for new
   exported surface.
4. **After the push, verify from outside: `GOWORK=off go run -C tools ./consumergate --tag
   gpu/vX.Y.Z`.** The in-tree gates cannot see a broken `require` — the replaces mask it —
   so this is the only check that sees what a consumer sees. CI's `consumer-resolution` job
   runs the same thing on the pushed tag; run it locally if you want the answer before CI does.

A local `pre-push` hook that runs `go run -C tools ./gpudevice` when a `gpu/v*` tag is being
pushed would enforce this by construction and is welcome — but one machine can only ever
produce one of the two verdicts, so the checklist above is the portable minimum: it turns a habit into a step
someone can visibly fail to complete.

## Backend submodules — the eight that have never been tagged

`gpu/anncuda`, `gpu/annmetal`, `gpu/enccuda`, `gpu/encmetal`, `gpu/qwencuda`,
`gpu/qwenmetal`, `gpu/visioncuda`, `gpu/visionmetal` are separate Go modules and **none
of them has ever carried a tag**. Until 2026-08-12 nothing above said so, and the
consequence was invisible from inside the repo: everything here builds, because each of
them carries `replace` directives into the tree.

**A consumer sees something different, and it was verified rather than reasoned about.**
An external module importing `aikit/gpu/anncuda`:

1. **resolves the backend fine** — Go synthesizes a pseudo-version from the latest
   commit, so the absence of a tag is not itself the problem;
2. **then fails on its dependency**: `require github.com/townsendmerino/aikit/gpu v0.0.0`
   is a version that does not exist —
   `reading .../gpu/@v/v0.0.0.zip: 404 Not Found`.

The `replace` lines are NOT the fault. Go ignores replace directives from a dependency's
go.mod, which is correct and is what the 404 proves — the build tried to fetch
`gpu@v0.0.0` from the proxy rather than following the replace. Keeping them for local
development is fine.

A consumer *can* work around it, which is why this went unnoticed: adding an explicit
`require github.com/townsendmerino/aikit/gpu v0.27.0` makes MVS pick the higher version
and resolution succeeds. **It then fails to compile** —
`dev.SMCount undefined (type *gpu.Device has no field or method SMCount)` — because the
backend's code needs `gpu` surface newer than any tag. That is the real shape of the
problem: `v0.0.0` does not merely fail, it forces every consumer to guess a version, and
a wrong guess surfaces as a compile error inside someone else's dependency.

### Fixing it, in this order

Each layer's `require` must resolve before the layer above it can be tagged, so the
order is not optional:

1. **Root** — tag `vX.Y.Z`. Nothing here depends on the rest of the repo.
2. **`gpu`** — tag `gpu/vX.Y.Z`. Also depends on nothing in-repo (`gpu/go.mod` has no
   replaces).
3. **The eight** — set `require github.com/townsendmerino/aikit <root tag>` and
   `require github.com/townsendmerino/aikit/gpu <gpu tag>` in each go.mod, keep the
   replaces, commit, then tag `gpu/anncuda/vX.Y.Z` and so on.

Step 3 cannot be done before steps 1 and 2, because the versions it names have to exist.
The window in between is not a regression: these modules do not resolve for consumers
today either.

Verify from outside afterwards, not from inside the repo — the replaces make an in-tree
check meaningless. That check is now a Go tool: **`GOWORK=off go run -C tools ./consumergate
--tag <tag>`** resolves and builds the tag's module from a scratch module with no replaces,
`GOWORK=off`, and the public proxy as the only source. The `consumer-resolution` CI job
(`.github/workflows/consumer.yml`) runs it on every release tag, and a weekly job runs the
whole published set at `@latest` — so the "does not resolve for consumers" class is a
standing watch, not a thing to remember on release day. A red run cannot unpublish an
immutable tag; it shortens detection to minutes, and the fix is a follow-up tag (above).

### How to tell whether a fix is already released

**Ask `git tag --contains <commit>`, per commit. Do not reason from "commits since
`<tag>`".** The second measures *distance* and says nothing about *release state* — a
commit can sit many commits back and still be unreleased, or be one commit back and
already tagged. Only containment answers the question:

```sh
git tag --list 'gpu/v*' --contains <commit> --sort=v:refname | head -1   # first tag shipping it
grep -rn 'aikit/gpu ' ../goinfer/*/go.mod                               # what consumers require
```

Those two lines together answer "is the consumer blocked?" — which is the only
question that justifies an out-of-cycle tag.

### Recorded non-decision: 2026-08-11, no tag cut

A `gpu/v0.28.0` was proposed to unblock goinfer's C-09 and M-11 audit findings. **It
was not cut, because the premise did not survive checking:**

| commit | first `gpu/` tag containing it |
|---|---|
| `cea19ab` C-09 `Encoder.Err()` | `gpu/v0.26.0` |
| `6bb28fc` C-09 latch at WaitDone/End | `gpu/v0.26.1` |
| `4642b7c` M-11 `MaxThreadgroupMemoryLength()` | `gpu/v0.27.0` — it *is* that tag |

goinfer's `cuda/go.mod` and `metal/go.mod` already require `v0.27.0`, so both findings
were shipped and pinned before the question was asked. Nothing was blocked.

What was actually unreleased: two new `_test.go` files and comment-only edits to three
`.cu` sources and one `.go` file — **the PTX byte-identical**, as the new
reproducibility test proves. No exported surface changed, so the rule above would have
made it a **patch (`v0.27.1`)**, not a minor.

It was declined even at patch level. A Go module tag is **immutable once the proxy
fetches it**, and this one would have announced a property no consumer receives: a
version whose only content is assurance machinery that runs in *this* repo. The three
gates and the first CI job ride along with the next release that has a substantive
reason of its own — likely **v1.0**, where "the GPU modules acquired their first gates
and their first CI" is a real line item rather than the whole changelog.

The general form: *a release needs a reason a consumer can receive.* Test coverage,
lint rules and CI are properties of the repository, not of the artifact.

### Why CI is not the enforcement point (revisit at v1.0)

`main` has **no branch protection and no rulesets**, so no aikit CI check is required —
not `gpu-kernels`, not `core`, not any of them. That is a deliberate choice, not an
oversight:

- Required status checks effectively force **PR-only merges**, because a direct push
  cannot carry passing checks for a commit that does not exist yet. Both development
  boxes push straight to `main` and rebase on each other; that workflow would end.
- The friction buys protection against a threat model aikit does not have. There are no
  outside contributors — every push is a maintainer's.
- The real risk is **a maintainer not running the check**, and protection relocates that
  risk rather than solving it: the same person can merge a PR whose checks they did not
  read. Putting the gate in the tagging ritual puts it where the risk actually is.

Revisit when external contributors become plausible, i.e. at v1.0. At that point
required checks are the right tool, because the threat model will have changed.
