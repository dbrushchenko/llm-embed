//go:build linux && amd64 && !cgo

// Dynamic linking to libdl.so.2 for dlopen/dlsym without CGO.
// Adapted from ebitengine/purego (Apache-2.0 license).
package ggml

//go:cgo_import_dynamic purego_dlopen dlopen "libdl.so.2"
//go:cgo_import_dynamic purego_dlsym dlsym "libdl.so.2"
//go:cgo_import_dynamic purego_dlerror dlerror "libdl.so.2"
//go:cgo_import_dynamic _ _ "libdl.so.2"
