//go:build linux && amd64 && !cgo

// Dynamic linking to libdl.so.2 for dlopen/dlsym without CGO.
package ggml

//go:cgo_import_dynamic dlopen_sym dlopen "libdl.so.2"
//go:cgo_import_dynamic dlsym_sym dlsym "libdl.so.2"
//go:cgo_import_dynamic dlerror_sym dlerror "libdl.so.2"
//go:cgo_import_dynamic _ _ "libdl.so.2"
