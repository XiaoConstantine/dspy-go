package experimental_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const experimentalImportPrefix = "github.com/XiaoConstantine/dspy-go/pkg/experimental"

// TestStablePackagesDoNotImportExperimental enforces the one-way incubation
// boundary: experimental packages may use stable packages, never the reverse.
func TestStablePackagesDoNotImportExperimental(t *testing.T) {
	pkgRoot, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	experimentalRoot := filepath.Join(pkgRoot, "experimental")

	err = filepath.WalkDir(pkgRoot, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path == experimentalRoot {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(entry.Name(), ".go") {
			return nil
		}

		file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, declaration := range file.Decls {
			importDeclaration, ok := declaration.(*ast.GenDecl)
			if !ok || importDeclaration.Tok != token.IMPORT {
				continue
			}
			for _, spec := range importDeclaration.Specs {
				importSpec := spec.(*ast.ImportSpec)
				importPath, err := strconv.Unquote(importSpec.Path.Value)
				if err != nil {
					return err
				}
				if importPath == experimentalImportPrefix || strings.HasPrefix(importPath, experimentalImportPrefix+"/") {
					relative, _ := filepath.Rel(pkgRoot, path)
					t.Errorf("stable package file %s imports experimental package %q", relative, importPath)
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
