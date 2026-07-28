module github.com/townsendmerino/aikit/gpu/qwencuda

go 1.26.5

require (
	github.com/townsendmerino/aikit v1.12.0
	github.com/townsendmerino/aikit/gpu v0.0.0
)

require (
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/eitamring/gocudrv v0.2.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.40.0 // indirect
)

replace github.com/townsendmerino/aikit => ../../

replace github.com/townsendmerino/aikit/gpu => ../
