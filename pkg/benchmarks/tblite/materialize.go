package tblite

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
)

const (
	maxArchiveFileSizeBytes  int64 = 100 << 20
	maxArchiveTotalSizeBytes int64 = 512 << 20
)

// MaterializedTask describes a TBLite task expanded onto disk.
type MaterializedTask struct {
	Task            datasets.TBLiteTask
	RootDir         string
	EnvironmentDir  string
	TestsDir        string
	TestScriptPath  string
	InstructionPath string
}

// MaterializeTask decodes the task archives into a runnable directory structure.
func MaterializeTask(task datasets.TBLiteTask, rootDir string) (*MaterializedTask, error) {
	task = task.Normalize()
	if err := datasets.ValidateTBLiteTaskName(task.TaskName); err != nil {
		return nil, err
	}
	if rootDir == "" {
		return nil, fmt.Errorf("root dir is required")
	}

	taskRoot := filepath.Join(rootDir, task.TaskName)
	envDir := filepath.Join(taskRoot, "environment")
	testsDir := filepath.Join(taskRoot, "tests")
	instructionPath := filepath.Join(taskRoot, "instruction.txt")
	testScriptPath := filepath.Join(taskRoot, "test.sh")

	for _, dir := range []string{taskRoot, envDir, testsDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("create dir %s: %w", dir, err)
		}
	}

	if err := extractArchive(task.EnvironmentTar, envDir); err != nil {
		return nil, fmt.Errorf("extract environment: %w", err)
	}
	if err := extractArchive(task.TestsTar, testsDir); err != nil {
		return nil, fmt.Errorf("extract tests: %w", err)
	}
	if err := os.WriteFile(instructionPath, []byte(task.Instruction), 0o644); err != nil {
		return nil, fmt.Errorf("write instruction: %w", err)
	}
	if err := os.WriteFile(testScriptPath, []byte(task.TestScript), 0o755); err != nil {
		return nil, fmt.Errorf("write test script: %w", err)
	}

	return &MaterializedTask{
		Task:            task,
		RootDir:         taskRoot,
		EnvironmentDir:  envDir,
		TestsDir:        testsDir,
		TestScriptPath:  testScriptPath,
		InstructionPath: instructionPath,
	}, nil
}

func extractArchive(payload string, targetDir string) error {
	return extractArchiveWithLimits(payload, targetDir, maxArchiveFileSizeBytes, maxArchiveTotalSizeBytes)
}

func extractArchiveWithLimits(payload string, targetDir string, maxFileSizeBytes int64, maxTotalSizeBytes int64) error {
	archiveBytes, err := decodeArchive(payload)
	if err != nil {
		return err
	}
	if len(archiveBytes) == 0 {
		return nil
	}

	reader, err := newTarReader(archiveBytes)
	if err != nil {
		return err
	}

	var totalExtractedBytes int64
	for {
		header, err := reader.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read tar entry: %w", err)
		}

		if err := validateTarEntry(header, maxFileSizeBytes, maxTotalSizeBytes, &totalExtractedBytes); err != nil {
			return err
		}

		entryReader := io.Reader(reader)
		if header.Typeflag == tar.TypeReg || header.Typeflag == tar.TypeRegA {
			entryReader = io.LimitReader(reader, header.Size)
		}

		if err := writeTarEntry(targetDir, header, entryReader); err != nil {
			return err
		}
	}
}

func decodeArchive(payload string) ([]byte, error) {
	payload = strings.TrimSpace(payload)
	if payload == "" {
		return nil, nil
	}
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil, fmt.Errorf("decode archive: %w", err)
	}
	return decoded, nil
}

func newTarReader(data []byte) (*tar.Reader, error) {
	buffer := bytes.NewReader(data)
	gzipReader, err := gzip.NewReader(buffer)
	if err == nil {
		return tar.NewReader(gzipReader), nil
	}

	buffer = bytes.NewReader(data)
	return tar.NewReader(buffer), nil
}

func writeTarEntry(root string, header *tar.Header, reader io.Reader) error {
	name := filepath.Clean(strings.TrimPrefix(header.Name, "./"))
	if name == "." || name == "" {
		return nil
	}
	if strings.HasPrefix(name, "..") {
		return fmt.Errorf("refusing to extract path traversal entry %q", header.Name)
	}

	destination := filepath.Join(root, name)
	if !strings.HasPrefix(destination, filepath.Clean(root)+string(os.PathSeparator)) && filepath.Clean(destination) != filepath.Clean(root) {
		return fmt.Errorf("refusing to extract outside root: %q", header.Name)
	}

	switch header.Typeflag {
	case tar.TypeDir:
		return os.MkdirAll(destination, 0o755)
	case tar.TypeReg, tar.TypeRegA:
		if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
			return fmt.Errorf("create parent dir: %w", err)
		}
		file, err := os.OpenFile(destination, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(header.Mode))
		if err != nil {
			return fmt.Errorf("open file %s: %w", destination, err)
		}
		defer file.Close()
		if _, err := io.Copy(file, reader); err != nil {
			return fmt.Errorf("write file %s: %w", destination, err)
		}
		return nil
	case tar.TypeSymlink, tar.TypeLink:
		return fmt.Errorf("refusing to extract link entry %q", header.Name)
	default:
		return nil
	}
}

func validateTarEntry(header *tar.Header, maxFileSizeBytes int64, maxTotalSizeBytes int64, totalExtractedBytes *int64) error {
	if header == nil {
		return fmt.Errorf("tar header is required")
	}
	switch header.Typeflag {
	case tar.TypeSymlink, tar.TypeLink:
		return fmt.Errorf("refusing to extract link entry %q", header.Name)
	case tar.TypeReg, tar.TypeRegA:
		if header.Size < 0 {
			return fmt.Errorf("refusing to extract negative-sized entry %q", header.Name)
		}
		if maxFileSizeBytes > 0 && header.Size > maxFileSizeBytes {
			return fmt.Errorf("refusing to extract oversized entry %q (%d bytes > %d byte limit)", header.Name, header.Size, maxFileSizeBytes)
		}
		if totalExtractedBytes != nil {
			*totalExtractedBytes += header.Size
			if maxTotalSizeBytes > 0 && *totalExtractedBytes > maxTotalSizeBytes {
				return fmt.Errorf("refusing to extract archive exceeding %d byte total limit", maxTotalSizeBytes)
			}
		}
	}
	return nil
}
