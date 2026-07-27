module github.com/townsendmerino/aikit/gpu/annmetal

go 1.26.5

require (
	github.com/townsendmerino/aikit v1.11.0
	github.com/townsendmerino/aikit/gpu v0.0.0
)

require (
	github.com/ebitengine/purego v0.10.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/townsendmerino/aikit => ../../

replace github.com/townsendmerino/aikit/gpu => ../
