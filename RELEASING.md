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
3. Push the prep commit, wait for **root CI green**, then push the tag `vX.Y.Z`.
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

1. On a Mac with a Metal GPU (and, for the CUDA half, a Linux/NVIDIA box if that path
   changed): `cd gpu && CGO_ENABLED=0 go test ./...`. The Metal parity gates
   (`metal_vit_test.go`) and the compile guards (`metal_precise_test.go`) must pass;
   CUDA tests skip cleanly without an NVIDIA device.
2. `gh auth switch --user townsendmerino` before pushing (a second account on this
   machine 403s on `townsendmerino` repos), and author commits as
   `townsendmerino@gmail.com`.
3. Tag **annotated**, with the environment in the message, so the "tested on" record
   travels with the tag:
   ```
   git tag -a gpu/vX.Y.Z -m "gpu vX.Y.Z — <what changed>
   tested: <machine>, macOS <version>, gpu/ suite green"
   git push origin gpu/vX.Y.Z
   ```
   Bump the **patch** for a fix/hardening with no API change, the **minor** for new
   exported surface.

A local `pre-push` hook that runs `cd gpu && go test ./...` when a `gpu/v*` tag is being
pushed would enforce this by construction and is welcome — but it only works on a Mac
with a GPU, so the checklist above is the portable minimum: it turns a habit into a step
someone can visibly fail to complete.
