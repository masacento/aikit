package mmap

import "container/list"

// EvictPolicy selects which resident member a SpanCache releases when a fault pushes
// it over budget. The right choice depends on the caller's DEMAND SIGNAL, and the two
// signals in the kit want opposite policies:
//
//   - EvictMostRecent (scan-resistant, the default). For a cyclic SCAN — an ANN paged
//     query walking blocks 0,1,2,… every pass — plain LRU is pathological: the block
//     evicted to make room is exactly the one wanted next round, so the cache hits 0%
//     even at a 63/64 budget. Evicting the most-recent instead pins a stable prefix,
//     recovering ~budget/working-set of the touches (perf-campaign item 9, 6c0483f).
//
//   - EvictLeastRecent (classic LRU tail, frequency-aware). For a skewed-FREQUENCY
//     signal — a MoE router whose hottest 10% of experts absorb ~72% of top-k picks —
//     the hot set is what must stay resident. Evict-most-recent throws the hot set out
//     the instant anything else is touched and pins the cold prefix; on a real 35B-A3B
//     trace that cost up to 51 pp of hit rate vs the LRU tail at interactive budgets.
//     A frequency-skewed pager (expertPager) must use this.
type EvictPolicy uint8

const (
	EvictMostRecent  EvictPolicy = iota // scan-resistant (default): release the most-recently-touched OTHER member
	EvictLeastRecent                    // classic LRU: release the least-recently-touched (budget tail)
)

// SpanCache bounds the resident RAM of a read-only mapping by paging spans of it in
// and out under a byte budget. It is the generic, demand-signal-agnostic core of a
// weight/index pager: the caller registers each member's page-aligned spans with
// Add and calls Touch(key) when a member is needed now. Touch faults the member in
// (Advise WILLNEED) and, if that pushes resident bytes over budget, releases members
// (Advise DONTNEED) per the configured EvictPolicy — the most-recently-touched other
// member (scan-resistant, the default) or the least-recently-touched (classic LRU) —
// until back under it.
//
// Releasing is lossless: a MapReadOnly mapping is read-only and file-backed, so an
// evicted-then-reused member merely re-faults from the file — its bytes are
// identical, only the cold-miss fault costs anything. So a budget-paged traversal
// returns the same results as a fully-resident one; the cap trades faults for RAM.
//
// SpanCache holds NO model-specific logic. It does not know what a member is or when
// to Touch it — the demand signal (a MoE router's top-k, an ANN query's scanned
// blocks, a layer-order prefetch) lives entirely in the caller. The key type K is
// whatever the caller uses to identify a member (a tensor pointer, a block index).
//
// Not goroutine-safe — one traversal touches it at a time, like a KV cache.
type SpanCache[K comparable] struct {
	budget   int64                    // resident-bytes cap; ≤ 0 means unbounded (never evict)
	resident int64                    // bytes currently held resident (our accounting)
	spans    map[K][][]byte           // page-aligned spans of the mapping, per member
	bytes    map[K]int64              // resident bytes per member (Σ aligned span lens)
	lru      *list.List               // K, most-recently-touched at front
	pos      map[K]*list.Element      // resident membership + O(1) promotion
	advise   func([]byte, bool) error // residency hint; Advise in production, a fake in tests
	policy   EvictPolicy              // which member to release over budget (see EvictPolicy)

	hits, misses, evictions int64
	advisedBytes            int64 // cumulative bytes passed to advise(_, true) (WILLNEED) over all Touch calls
}

// NewSpanCache returns a cache that caps resident registered spans at budget bytes,
// with the default scan-resistant EvictMostRecent policy. A budget ≤ 0 means
// unbounded: members are still tracked and prefetched on Touch, but nothing is ever
// evicted. Register members with Add before touching them. Callers whose demand
// signal is skewed-frequency rather than a scan (a MoE expert pager) should use
// NewSpanCacheWithPolicy(budget, EvictLeastRecent) — see EvictPolicy.
func NewSpanCache[K comparable](budget int64) *SpanCache[K] {
	return NewSpanCacheWithPolicy[K](budget, EvictMostRecent)
}

// NewSpanCacheWithPolicy is NewSpanCache with an explicit eviction policy. Pick the
// policy from your demand signal, not by default: EvictMostRecent for a cyclic scan,
// EvictLeastRecent for a skewed-frequency access pattern (see EvictPolicy for why the
// wrong one silently regresses hit rate).
func NewSpanCacheWithPolicy[K comparable](budget int64, policy EvictPolicy) *SpanCache[K] {
	return &SpanCache[K]{
		budget: budget,
		spans:  map[K][][]byte{},
		bytes:  map[K]int64{},
		lru:    list.New(),
		pos:    map[K]*list.Element{},
		advise: Advise,
		policy: policy,
	}
}

// Add registers a member's spans under key. spans should be page-aligned (see
// PageAlignedInterior) so eviction of one member never disturbs another's pages;
// empty spans are dropped. A member starts non-resident — call Touch to fault it in.
// Re-adding a key that was already registered is ignored (the first registration
// wins); Add a distinct key per member.
func (c *SpanCache[K]) Add(key K, spans [][]byte) {
	if _, ok := c.spans[key]; ok {
		return
	}
	var kept [][]byte
	var n int64
	for _, s := range spans {
		if len(s) == 0 {
			continue
		}
		kept = append(kept, s)
		n += int64(len(s))
	}
	if n == 0 {
		return // nothing mapping-backed to page for this member
	}
	c.spans[key] = kept
	c.bytes[key] = n
}

// Touch records that key is needed now: it becomes most-recently-touched and, if it
// wasn't resident, is faulted in (Advise WILLNEED) and members are released per the
// EvictPolicy to stay within budget. A no-op for keys that were never Added.
func (c *SpanCache[K]) Touch(key K) {
	spans, managed := c.spans[key]
	if !managed {
		return
	}
	if el, ok := c.pos[key]; ok {
		c.lru.MoveToFront(el)
		c.hits++
		return
	}
	c.misses++
	for _, s := range spans {
		_ = c.advise(s, true) //nolint:errcheck // WILLNEED: hint the fault we're about to take (advisory)
		c.advisedBytes += int64(len(s))
	}
	el := c.lru.PushFront(key)
	c.pos[key] = el
	c.resident += c.bytes[key]
	// Release members over budget per the configured policy (see EvictPolicy). The
	// just-touched member is never the victim: it was faulted in one line ago and the
	// caller is about to read it — under EvictMostRecent it is skipped explicitly;
	// under EvictLeastRecent it sits at the front and the tail is taken.
	for c.budget > 0 && c.resident > c.budget && c.lru.Len() > 1 {
		var victimEl *list.Element
		if c.policy == EvictLeastRecent {
			victimEl = c.lru.Back() // classic LRU tail (never el, which is at the front)
		} else {
			victimEl = c.lru.Front() // scan-resistant: the most-recent OTHER member
			if victimEl == el {
				victimEl = victimEl.Next()
			}
		}
		if victimEl == nil {
			break
		}
		victim := victimEl.Value.(K)
		c.lru.Remove(victimEl)
		delete(c.pos, victim)
		c.resident -= c.bytes[victim]
		c.evictions++
		for _, s := range c.spans[victim] {
			_ = c.advise(s, false) //nolint:errcheck // DONTNEED: release the victim's pages (advisory)
		}
	}
}

// Resident reports the bytes the cache currently holds resident (its own accounting,
// Σ aligned span lengths of touched-and-not-evicted members). It never exceeds
// Budget once a member larger than the budget isn't in play.
func (c *SpanCache[K]) Resident() int64 { return c.resident }

// Budget reports the resident-bytes cap (≤ 0 means unbounded).
func (c *SpanCache[K]) Budget() int64 { return c.budget }

// Registered reports the total bytes of all Added members (resident or not) — the
// full footprint were nothing evicted. Useful for choosing a budget and for the
// load banner.
func (c *SpanCache[K]) Registered() int64 {
	var total int64
	for _, n := range c.bytes {
		total += n
	}
	return total
}

// Stats returns cumulative (hits, misses, evictions) over all Touch calls. A
// non-zero eviction count means the budget was actually enforced (the LRU tail was
// released), as opposed to mere cold-start misses.
func (c *SpanCache[K]) Stats() (hits, misses, evictions int64) {
	return c.hits, c.misses, c.evictions
}

// AdvisedBytes reports the cumulative bytes passed to the WILLNEED residency hint
// over every miss across all Touch calls — what THIS cache asked the OS to fetch,
// independent of what else the machine's disk did meanwhile. A durable, contamination-
// proof I/O check a benchmark can assert on directly (bytes advised / tokens generated
// should track the expected per-token working set), unlike an external tool (iostat,
// /proc/diskstats) that counts physical reads shared with every other process on the
// box. Divide by Misses() (via Stats) for bytes-per-miss; the fix this exists to catch
// is exactly a member whose registered spans are bigger than they need to be — kind-4
// briefly registered a tensor's canonical AND row4 spans together, doubling this number
// for no read the kernel ever performed (docs/task-zeno-compare.md's "At-scale
// acceptance run" in goinfer, the forcing function for this method).
func (c *SpanCache[K]) AdvisedBytes() int64 { return c.advisedBytes }
