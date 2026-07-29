package bench

import (
	"fmt"
	"sort"
	"strings"
)

// report.go GENERATES docs/BENCH-gpu-results.md from a set of records (report.go never
// hand-typed, per docs/BENCH-gpu.md). Two views, and deliberately NOT a third:
//
//   - PER-MACHINE tables — the primary, apples-to-apples view. One table per box, a true
//     same-box cpu-vs-that-box's-GPU comparison. Absolute throughput compares only within a box.
//   - A NORMALIZED cross-platform summary — the only honest all-backends view: speedup over the
//     CPU each GPU actually ships next to. Absolute ms across machines are never put in adjacent
//     columns, because the CPU baselines are two different chips.
//
// The join key is (workload × precision × shape); a GPU row pairs with the cpu-simd row of the
// same key ON THE SAME MACHINE.

// shapeStr renders only the Shape fields a workload set — compact and stable for grouping/sort.
func shapeStr(s Shape) string {
	var p []string
	add := func(k string, v int) {
		if v != 0 {
			p = append(p, fmt.Sprintf("%s=%s", k, humanInt(v)))
		}
	}
	add("N", s.N)
	add("dim", s.Dim)
	add("batch", s.Batch)
	add("k", s.K)
	add("seq", s.Seq)
	add("patches", s.Patches)
	return strings.Join(p, " ")
}

func humanInt(v int) string {
	switch {
	case v >= 1_000_000 && v%1_000_000 == 0:
		return fmt.Sprintf("%dM", v/1_000_000)
	case v >= 1_000 && v%1_000 == 0:
		return fmt.Sprintf("%dk", v/1_000)
	default:
		return fmt.Sprintf("%d", v)
	}
}

// logicalKey identifies a comparable measurement across machines (everything but the backend/box).
func logicalKey(r Record) string {
	return r.Workload + "|" + r.Precision + "|" + shapeStr(r.Shape)
}

func machineKey(r Record) string {
	if r.Device.Machine != "" {
		return r.Device.Machine
	}
	return r.Device.GOARCH
}

// lessRecord orders rows the way a reader scans them — by workload, precision, then the shape
// fields NUMERICALLY (so batch 8 precedes 64 precedes 256, not the lexical 1/256/64/8).
func lessRecord(a, b Record) bool {
	if a.Workload != b.Workload {
		return a.Workload < b.Workload
	}
	if a.Precision != b.Precision {
		return a.Precision < b.Precision
	}
	s, t := a.Shape, b.Shape
	for _, p := range [][2]int{{s.N, t.N}, {s.Dim, t.Dim}, {s.Batch, t.Batch}, {s.K, t.K}, {s.Seq, t.Seq}, {s.Patches, t.Patches}} {
		if p[0] != p[1] {
			return p[0] < p[1]
		}
	}
	return false
}

func fmtThroughput(r Record) string {
	if r.Throughput == 0 {
		return "—"
	}
	unit := r.ThroughputUnit
	if unit == "" {
		unit = "/s"
	}
	return fmt.Sprintf("%s %s", humanFloat(r.Throughput), unit)
}

func humanFloat(v float64) string {
	switch {
	case v >= 1e6:
		return fmt.Sprintf("%.2fM", v/1e6)
	case v >= 1e3:
		return fmt.Sprintf("%.1fk", v/1e3)
	default:
		return fmt.Sprintf("%.1f", v)
	}
}

func recallStr(q Quality) string {
	if q.RecallAtK == nil {
		return "—"
	}
	return fmt.Sprintf("%.4f", *q.RecallAtK)
}

func parityStr(q Quality) string {
	if q.ParityOK {
		return "✅"
	}
	return "❌ FAIL"
}

// speedup returns gpu.Throughput / cpu.Throughput (recomputed from the throughputs so it can
// never disagree with the numbers in the same row), 0 if either is missing.
func speedup(gpu, cpu Record) float64 {
	if cpu.Throughput == 0 || gpu.Throughput == 0 {
		return 0
	}
	return gpu.Throughput / cpu.Throughput
}

// Report renders the full results document from records. Deterministic: everything is sorted, and
// the only volatile datum embedded is the aikit commit (not a wall-clock time), so re-generating
// unchanged records yields a byte-identical doc.
func Report(recs []Record) string {
	var b strings.Builder
	commit := "unknown"
	for _, r := range recs {
		if r.Meta.AikitCommit != "" {
			commit = r.Meta.AikitCommit
			break
		}
	}

	b.WriteString("# GPU benchmark results\n\n")
	b.WriteString("> **GENERATED** by `bench report` from `records.jsonl` — do not edit by hand.\n")
	b.WriteString("> Re-generate after each periodic run (`docs/BENCH-gpu.md`). Cross-machine\n")
	b.WriteString("> absolute numbers are never placed in adjacent columns — the CPU baselines are\n")
	b.WriteString("> different chips; compare within a machine, or via the normalized summary.\n\n")

	if len(recs) == 0 {
		b.WriteString("_No runs recorded yet._ Produce `records.jsonl` with the device-gated GPU\n")
		b.WriteString("harnesses (`gpu/annmetal` on Apple, `gpu/anncuda` on NVIDIA), then run\n")
		b.WriteString("`go run ./bench/cmd/benchreport records.jsonl > docs/BENCH-gpu-results.md`.\n")
		return b.String()
	}
	fmt.Fprintf(&b, "aikit `%s` · %d records · ", commit, len(recs))

	// group: machine -> logicalKey -> backend -> Record; repByKey holds one representative per
	// logicalKey so rows sort by shape numerically (not lexically).
	byMachine := map[string]map[string]map[string]Record{}
	machines := []string{}
	logicalOrder := []string{}
	seenLogical := map[string]bool{}
	repByKey := map[string]Record{}
	for _, r := range recs {
		mk := machineKey(r)
		if byMachine[mk] == nil {
			byMachine[mk] = map[string]map[string]Record{}
			machines = append(machines, mk)
		}
		lk := logicalKey(r)
		if byMachine[mk][lk] == nil {
			byMachine[mk][lk] = map[string]Record{}
		}
		byMachine[mk][lk][r.Backend] = r
		if _, ok := repByKey[lk]; !ok {
			repByKey[lk] = r
		}
		if !seenLogical[lk] {
			seenLogical[lk] = true
			logicalOrder = append(logicalOrder, lk)
		}
	}
	sort.Strings(machines)
	sort.Slice(logicalOrder, func(i, j int) bool { return lessRecord(repByKey[logicalOrder[i]], repByKey[logicalOrder[j]]) })
	fmt.Fprintf(&b, "%d machine(s)\n\n", len(machines))

	// --- Per-machine tables ---
	b.WriteString("## Per-machine tables (apples-to-apples, same box)\n\n")
	for _, mk := range machines {
		rows := byMachine[mk]
		// Build the device header DETERMINISTICALLY (map iteration order is random): Chip from a
		// cpu-simd record, GPU/family from the gpu record, GOARCH from either. The gpu backend is
		// the lexically-smallest non-cpu backend on this box (there is normally exactly one).
		var dev Device
		gpuBackend := ""
		for _, byBk := range rows {
			if c, ok := byBk["cpu-simd"]; ok {
				if dev.Chip == "" {
					dev.Chip = c.Device.Chip
				}
				if dev.GOARCH == "" {
					dev.GOARCH = c.Device.GOARCH
				}
			}
			for bk, r := range byBk {
				if bk == "cpu-simd" {
					continue
				}
				if gpuBackend == "" || bk < gpuBackend {
					gpuBackend = bk
					dev.GPU, dev.SMorFamily, dev.Driver = r.Device.GPU, r.Device.SMorFamily, r.Device.Driver
				}
				if dev.GOARCH == "" {
					dev.GOARCH = r.Device.GOARCH
				}
			}
		}
		fmt.Fprintf(&b, "### %s\n\n", machineHeader(mk, dev, gpuBackend))
		gcol := gpuBackend
		if gcol == "" {
			gcol = "gpu"
		}
		fmt.Fprintf(&b, "| workload | shape | precision | cpu-%s q/s | %s q/s | speedup | recall@k | parity |\n", dev.GOARCH, gcol)
		b.WriteString("|---|---|---|--:|--:|--:|--:|:--:|\n")
		lks := make([]string, 0, len(rows))
		for k := range rows {
			lks = append(lks, k)
		}
		sort.Slice(lks, func(i, j int) bool { return lessRecord(repByKey[lks[i]], repByKey[lks[j]]) })
		for _, lk := range lks {
			byBk := rows[lk]
			cpu := byBk["cpu-simd"]
			gpu := pickGPU(byBk)
			ref := cpu
			if ref.Workload == "" {
				ref = gpu
			}
			sp := "—"
			if s := speedup(gpu, cpu); s != 0 {
				sp = fmt.Sprintf("%.2f×", s)
			}
			fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %s | %s | %s |\n",
				ref.Workload, shapeStr(ref.Shape), ref.Precision,
				thr(cpu), thr(gpu), sp, recallStr(gpuOrCPU(gpu, cpu).Quality), parityStr(gpuOrCPU(gpu, cpu).Quality))
		}
		b.WriteString("\n")
	}

	// --- Normalized cross-platform summary ---
	b.WriteString("## Normalized cross-platform summary (speedup over each box's own CPU)\n\n")
	b.WriteString("The only honest all-backends view: absolute ms don't compare across machines, but\n")
	b.WriteString("*speedup over the CPU each GPU ships next to* does — the decision-relevant number.\n\n")
	// collect which gpu backends exist
	gpuBackends := gpuBackendsPresent(recs)
	hdr := "| workload | shape | precision |"
	sep := "|---|---|---|"
	for _, g := range gpuBackends {
		hdr += fmt.Sprintf(" %s ×vs-cpu |", g)
		sep += "--:|"
	}
	fmt.Fprintln(&b, hdr)
	fmt.Fprintln(&b, sep)
	// index: logicalKey -> gpuBackend -> speedup (from whatever machine ran it)
	spByKey := map[string]map[string]float64{}
	for _, byLK := range byMachine {
		for lk, byBk := range byLK {
			cpu := byBk["cpu-simd"]
			for bk, r := range byBk {
				if bk == "cpu-simd" {
					continue
				}
				if spByKey[lk] == nil {
					spByKey[lk] = map[string]float64{}
				}
				spByKey[lk][bk] = speedup(r, cpu)
			}
		}
	}
	for _, lk := range logicalOrder {
		r, ok := repByKey[lk]
		if !ok {
			continue
		}
		row := fmt.Sprintf("| %s | %s | %s |", r.Workload, shapeStr(r.Shape), r.Precision)
		for _, g := range gpuBackends {
			if s, ok := spByKey[lk][g]; ok && s != 0 {
				row += fmt.Sprintf(" %.2f× |", s)
			} else {
				row += " — |"
			}
		}
		fmt.Fprintln(&b, row)
	}
	b.WriteString("\n")

	// --- Dispatch thresholds ---
	b.WriteString("## Dispatch thresholds (the crossover — the input to backend dispatch)\n\n")
	writeThresholds(&b, byMachine)
	return b.String()
}

func machineHeader(mk string, d Device, gpuBackend string) string {
	parts := []string{mk}
	if d.Chip != "" {
		parts = append(parts, d.Chip)
	}
	if d.GOARCH != "" {
		parts = append(parts, d.GOARCH)
	}
	g := d.GPU
	if g != "" {
		if d.SMorFamily != "" {
			g += " (" + d.SMorFamily + ")"
		}
		parts = append(parts, "· GPU "+g)
	}
	return strings.Join(parts, " ")
}

func pickGPU(byBk map[string]Record) Record {
	for bk, r := range byBk {
		if bk != "cpu-simd" {
			return r
		}
	}
	return Record{}
}

func gpuOrCPU(gpu, cpu Record) Record {
	if gpu.Workload != "" {
		return gpu
	}
	return cpu
}

func thr(r Record) string {
	if r.Workload == "" {
		return "—"
	}
	return fmtThroughput(r)
}

func gpuBackendsPresent(recs []Record) []string {
	set := map[string]bool{}
	for _, r := range recs {
		if r.Backend != "cpu-simd" && r.Backend != "" {
			set[r.Backend] = true
		}
	}
	out := make([]string, 0, len(set))
	for g := range set {
		out = append(out, g)
	}
	sort.Strings(out)
	return out
}

// writeThresholds finds, per (machine, workload, precision, N), the smallest batch where the GPU
// overtakes the CPU — the value the ann.Backend dispatch keys off.
func writeThresholds(b *strings.Builder, byMachine map[string]map[string]map[string]Record) {
	type pt struct {
		n, batch int
		sp       float64
	}
	// machine -> "workload|precision|N" -> []pt
	found := false
	machines := make([]string, 0, len(byMachine))
	for mk := range byMachine {
		machines = append(machines, mk)
	}
	sort.Strings(machines)
	for _, mk := range machines {
		grp := map[string][]pt{}
		for _, byBk := range byMachine[mk] {
			cpu := byBk["cpu-simd"]
			gpu := pickGPU(byBk)
			if cpu.Workload == "" || gpu.Workload == "" {
				continue
			}
			if gpu.Shape.Batch == 0 { // only batch-swept workloads (ANN) have a crossover threshold
				continue
			}
			key := fmt.Sprintf("%s (%s, N=%s)", gpu.Workload, gpu.Precision, humanInt(gpu.Shape.N))
			grp[key] = append(grp[key], pt{gpu.Shape.N, gpu.Shape.Batch, speedup(gpu, cpu)})
		}
		keys := make([]string, 0, len(grp))
		for k := range grp {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			pts := grp[k]
			sort.Slice(pts, func(i, j int) bool { return pts[i].batch < pts[j].batch })
			cross := -1
			for _, p := range pts {
				if p.sp >= 1 {
					cross = p.batch
					break
				}
			}
			found = true
			if cross >= 0 {
				fmt.Fprintf(b, "- **%s/%s**: GPU overtakes CPU at **batch ≥ %d**.\n", mk, k, cross)
			} else {
				fmt.Fprintf(b, "- **%s/%s**: CPU wins at every measured batch (GPU never overtakes).\n", mk, k)
			}
		}
	}
	if !found {
		b.WriteString("_No paired cpu/gpu points to derive a threshold from._\n")
	}
}
