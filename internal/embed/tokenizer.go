package embed

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"
)

// WordPieceTokenizer implements tokenization for nomic-bert.
// Despite the name, nomic uses a SentencePiece-style unigram vocab with ▁ prefixes.
type WordPieceTokenizer struct {
	vocab    map[string]int // token -> id
	idToTok  []string       // id -> token
	maxToken int            // max single-token length in vocab
}

const sentencePiecePrefix = "▁" // U+2581

// LoadTokenizer reads the vocabulary from a GGUF file's metadata.
func LoadTokenizer(path string) (*WordPieceTokenizer, error) {
	tokens, err := readGGUFStringArray(path, "tokenizer.ggml.tokens")
	if err != nil {
		return nil, fmt.Errorf("embed: load tokenizer: %w", err)
	}

	vocab := make(map[string]int, len(tokens))
	maxLen := 0
	for i, tok := range tokens {
		vocab[tok] = i
		if len(tok) > maxLen {
			maxLen = len(tok)
		}
	}

	return &WordPieceTokenizer{
		vocab:    vocab,
		idToTok:  tokens,
		maxToken: maxLen,
	}, nil
}

// Encode tokenizes text into token IDs with [CLS] and [SEP].
// Returns tokenIDs and attentionMask (1 for real tokens, 0 for padding).
func (t *WordPieceTokenizer) Encode(text string, maxLen int) ([]int, []int) {
	// Normalize: lowercase
	text = strings.ToLower(text)

	// Split into words (on whitespace and punctuation)
	words := preTokenize(text)

	// Encode each word using greedy longest-match with ▁ prefix
	var ids []int
	ids = append(ids, 101) // [CLS]

	for _, word := range words {
		wordIDs := t.encodeWord(word)
		ids = append(ids, wordIDs...)
	}

	ids = append(ids, 102) // [SEP]

	// Truncate if needed
	if len(ids) > maxLen {
		ids = ids[:maxLen]
		ids[maxLen-1] = 102 // ensure [SEP] at end
	}

	// Attention mask: all 1s for actual tokens
	mask := make([]int, len(ids))
	for i := range mask {
		mask[i] = 1
	}

	return ids, mask
}

// encodeWord tokenizes a single word using greedy longest-match.
// The first piece gets a ▁ prefix, subsequent pieces don't.
func (t *WordPieceTokenizer) encodeWord(word string) []int {
	// Try the whole word with ▁ prefix first
	fullToken := sentencePiecePrefix + word
	if id, ok := t.vocab[fullToken]; ok {
		return []int{id}
	}

	// Greedy longest-match tokenization
	var ids []int
	start := 0
	for start < len(word) {
		end := len(word)
		// Limit to max token length
		maxEnd := start + t.maxToken
		if maxEnd < end {
			end = maxEnd
		}

		found := false
		for end > start {
			var candidate string
			if start == 0 {
				candidate = sentencePiecePrefix + word[start:end]
			} else {
				candidate = word[start:end]
			}

			if id, ok := t.vocab[candidate]; ok {
				ids = append(ids, id)
				start = end
				found = true
				break
			}
			end--
		}

		if !found {
			// Try single character
			if start == 0 {
				candidate := sentencePiecePrefix + word[start:start+1]
				if id, ok := t.vocab[candidate]; ok {
					ids = append(ids, id)
					start++
					continue
				}
			}
			// Single char without prefix
			candidate := word[start : start+1]
			if id, ok := t.vocab[candidate]; ok {
				ids = append(ids, id)
				start++
				continue
			}
			// Unknown token
			ids = append(ids, 100) // [UNK]
			start++
		}
	}
	return ids
}

// VocabSize returns the vocabulary size.
func (t *WordPieceTokenizer) VocabSize() int {
	return len(t.idToTok)
}

// preTokenize lowercases and splits text into word tokens.
// Splits on whitespace and punctuation (punctuation becomes separate tokens).
// Matches BERT's BasicTokenizer behavior.
func preTokenize(text string) []string {
	text = strings.ToLower(text)

	var words []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		} else if isBertPunct(r) {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
			words = append(words, string(r))
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}
	return words
}

// isBertPunct matches BERT's _is_punctuation: any non-alnum, non-space ASCII
// character, plus Unicode punctuation category.
func isBertPunct(r rune) bool {
	if (r >= '!' && r <= '/') || (r >= ':' && r <= '@') ||
		(r >= '[' && r <= '`') || (r >= '{' && r <= '~') {
		return true
	}
	return unicode.IsPunct(r) || isChinesePunct(r)
}

func isChinesePunct(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF)
}

// readGGUFStringArray re-reads a GGUF file to extract a string array by key name.
func readGGUFStringArray(path string, key string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Skip magic (4) + version (4) + numTensors (8) + numKV (8) = 24 bytes
	var header [24]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return nil, err
	}
	numKV := binary.LittleEndian.Uint64(header[16:24])

	// Scan KV pairs looking for our key
	for i := uint64(0); i < numKV; i++ {
		k, err := readGGUFStr(f)
		if err != nil {
			return nil, err
		}
		vtype, err := readU32(f)
		if err != nil {
			return nil, err
		}

		if k == key && vtype == 9 { // TypeArray
			return readGGUFStrArrayValue(f)
		}

		// Skip this value
		if err := skipGGUFValue(f, vtype); err != nil {
			return nil, fmt.Errorf("skipping KV %q (type %d): %w", k, vtype, err)
		}
	}

	return nil, fmt.Errorf("key %q not found in GGUF metadata", key)
}

func readGGUFStrArrayValue(f *os.File) ([]string, error) {
	elemType, err := readU32(f)
	if err != nil {
		return nil, err
	}
	if elemType != 8 { // TypeString
		return nil, fmt.Errorf("expected string array, got element type %d", elemType)
	}
	length, err := readU64(f)
	if err != nil {
		return nil, err
	}
	arr := make([]string, length)
	for i := uint64(0); i < length; i++ {
		s, err := readGGUFStr(f)
		if err != nil {
			return nil, err
		}
		arr[i] = s
	}
	return arr, nil
}

func readGGUFStr(f *os.File) (string, error) {
	length, err := readU64(f)
	if err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(f, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readU32(f *os.File) (uint32, error) {
	var buf [4]byte
	if _, err := io.ReadFull(f, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(buf[:]), nil
}

func readU64(f *os.File) (uint64, error) {
	var buf [8]byte
	if _, err := io.ReadFull(f, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}

func skipGGUFValue(f *os.File, vtype uint32) error {
	switch vtype {
	case 0, 1, 7: // uint8, int8, bool
		_, err := readBytes(f, 1)
		return err
	case 2, 3: // uint16, int16
		_, err := readBytes(f, 2)
		return err
	case 4, 5, 6: // uint32, int32, float32
		_, err := readBytes(f, 4)
		return err
	case 8: // string
		length, err := readU64(f)
		if err != nil {
			return err
		}
		_, err = readBytes(f, int(length))
		return err
	case 9: // array
		elemType, err := readU32(f)
		if err != nil {
			return err
		}
		length, err := readU64(f)
		if err != nil {
			return err
		}
		for i := uint64(0); i < length; i++ {
			if err := skipGGUFValue(f, elemType); err != nil {
				return err
			}
		}
		return nil
	case 10, 11, 12: // uint64, int64, float64
		_, err := readBytes(f, 8)
		return err
	default:
		return fmt.Errorf("unknown GGUF value type: %d", vtype)
	}
}

func readBytes(f *os.File, n int) ([]byte, error) {
	buf := make([]byte, n)
	_, err := io.ReadFull(f, buf)
	return buf, err
}
