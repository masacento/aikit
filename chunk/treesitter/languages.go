package treesitter

// languageGrammars maps aikit's canonical language names (from
// chunk/languages.go) to the grammar names registered in
// github.com/odvcencio/gotreesitter/grammars.
//
// Cover at minimum the 5 the regex chunker supports (python, go,
// typescript, java, rust) plus every language in semble's NDCG benchmark
// (cpp, scala, ruby, php, swift, kotlin, c, javascript, zig, elixir,
// haskell, lua, bash, csharp). gotreesitter ships them — the incremental
// cost is zero so we map them all.
//
// Later additions (dart, csharp, sql, html, css, scss, r) fit neither original bucket — they
// aren't among the regex chunker's 5 nor in semble's NDCG benchmark set;
// they're driven by real-world popularity. SQL is the clearest case: TIOBE
// #8 and 58.6% usage on Stack Overflow's 2025 survey (2025/2026 data).
// gotreesitter ships a real sql grammar (a direct 1:1 mapping), and the
// parse-timeout -> line-chunker fallback keeps a weak grammar graceful.
//
// Naming quirks:
//   - "csharp" ↔ gotreesitter "c_sharp"
//   - "shell" ↔ gotreesitter "bash" (bash is the only shell variant
//     in the benchmark; treating shell-as-bash gives us tree-sitter chunks
//     for the bash repos rather than falling through to the line chunker).
//   - "javascript" maps to the JS grammar directly so .js files get
//     proper AST chunking. (extLang historically routed .js →
//     "typescript" because the regex chunker reuses TS rules for JS; the
//     treesitter chunker handles each grammar natively and supports both
//     names.)
//
// gotreesitter exposes its grammars via grammars.DetectLanguageByName, so
// these strings are the same identifiers the upstream registry uses. A
// missing entry means: language not supported by this chunker; ChunkFile
// falls back to the line chunker, same as the regex chunker behavior.
//
// Unexported: no external package reads this directly (SupportedLanguages
// is the public accessor), so keeping it package-private removes the
// concurrent-read/external-mutation hazard flagged in the 2026-07-25 audit
// (docs/internal/archive/AUDIT-2026-07-25.md) without needing a mutex —
// nothing outside this package can reach it to mutate it.
var languageGrammars = map[string]string{
	"python":     "python",
	"go":         "go",
	"typescript": "typescript",
	"javascript": "javascript",
	"java":       "java",
	"rust":       "rust",
	"c":          "c",
	"cpp":        "cpp",
	"ruby":       "ruby",
	"php":        "php",
	"swift":      "swift",
	"kotlin":     "kotlin",
	"scala":      "scala",
	"haskell":    "haskell",
	"elixir":     "elixir",
	"lua":        "lua",
	"zig":        "zig",
	"dart":       "dart",
	"csharp":     "c_sharp", // OOM fixed upstream in gotreesitter v0.20.2 (bounded namespace recovery sub-parses); re-enabled.
	"sql":        "sql",     // popularity-driven (TIOBE #8; 58.6% on SO 2025) — direct 1:1 grammar; not in the NDCG set.
	"html":       "html",    // popularity-driven (SO 2025 61.9%); markup, not code-with-structure, but standalone .html carries searchable content.
	"css":        "css",     // popularity-driven (with html).
	"scss":       "scss",    // dedicated SCSS grammar, NOT the css alias: css errors on $vars/@mixin/@include (measured); .scss reroutes to this in chunk/languages.go extLang.
	"r":          "r",       // popularity-driven (TIOBE #9); direct 1:1 grammar. .r/.R routed via extLang.

	// Deliberately omitted:
	//   shell ("bash") — the gotreesitter v0.18.0 bash grammar parses
	//     real bash-it content extremely slowly: at chunkSize=1500 with
	//     a 1s per-parse timeout, ~39% of bash files time out and
	//     produce no AST chunks. Measured NDCG impact at v0.2.0 was
	//     −0.119 vs the regex chunker's line-fallback baseline. Falls
	//     back to the line chunker; tracked in DESIGN.md §10.
}
