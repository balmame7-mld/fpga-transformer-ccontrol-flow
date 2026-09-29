# =============================================================
# F1 — 36 real C control-flow statements (35 IF, 1 FOR)
# Extracted automatically from two public codebases, verified
# individually against the embedded tokenizer (no truncation).
#
# Sources:
#   - GNU coreutils, src/wc.c (Paul Rubin, David MacKenzie)
#     https://github.com/coreutils/coreutils/blob/master/src/wc.c
#     License: GPL-3.0-or-later
#   - cJSON.c (Dave Gamble and contributors)
#     https://github.com/DaveGamble/cJSON/blob/master/cJSON.c
#     License: MIT
#
# label: 0=IF, 1=WHILE, 2=FOR, 3=SWITCH
#
# Results (final corrected-corpus model, checkpoint
# Wq[0,0:8]=[6,-23,93,-20,28,-64,-23,60]):
#   - Pre tokenizer-identifier-fix : 30/36 (83.3%)
#   - Post tokenizer-identifier-fix: 28/36 (77.8%)
#   See paper Section VII-G for full discussion: the identifier-
#   classification fix (isalpha() -> proper C identifier regex)
#   produces verified-correct tokenization but does NOT improve
#   accuracy on this set -- zero of the original 6 failures were
#   corrected, and 2 new failures appeared. All residual failures
#   involve multi-word, underscore-separated real identifiers
#   absent in form from the synthetic training vocabulary,
#   regardless of whether they map to UNKNOWN or VARIABLE.
# =============================================================

real_examples = [
    ("if (print_linelength) printf (format_int, number_width, imaxtostr (linelength, buf));", 0),
    ("if (ferror (stdout)) write_error ();", 0),
    ("if (!use_avx512) use_avx512 = avx512_supported () ? 1 : -1;", 0),
    ("if (0 < use_avx512) return wc_lines_avx512 (fd);", 0),
    ("if (!use_avx2) use_avx2 = avx2_supported () ? 1 : -1;", 0),
    ("if (0 < use_avx2) return wc_lines_avx2 (fd);", 0),
    ("if (!use_neon) use_neon = neon_supported () ? 1 : -1;", 0),
    ("if (0 < use_neon) return wc_lines_neon (fd);", 0),
    ("if (MB_CUR_MAX > 1) { count_bytes = print_bytes; count_chars = print_chars; }", 0),
    ("if (!count_bytes || count_chars || print_lines || count_complicated) fdadvise (fd, 0, 0, FADVISE_SEQUENTIAL);", 0),
    ("if (current_pos < 0) ;", 0),
    ("if (bytes && 0 <= lseek (fd, bytes, SEEK_CUR)) skip_read = true;", 0),
    ("if (bytes_read < 0) { err = errno; break; }", 0),
    ("if (linepos > linelength) linelength = linepos;", 0),
    ("if (count_chars < print_chars) chars = bytes;", 0),
    ("if (total_mode != total_only) write_counts (lines, words, chars, bytes, linelength, file_x);", 0),
    ("if (linelength > max_line_length) max_line_length = linelength;", 0),
    ('if (err) error (0, err, "%s", quotef (file));', 0),
    ("if (!S_ISREG (fstatus[i].st.st_mode)) minimum_width = 7;", 0),
    ("if (width < minimum_width) width = minimum_width;", 0),
    ('if (streq (files_from, "-")) stream = stdin;', 0),
    ("if (!ai) xalloc_die ();", 0),
    ("if (total_mode == total_only) number_width = 1;", 0),
    ("if (skip_file) ok = false;", 0),
    ("if (! nfiles) fstatus[0].failed = 1;", 0),
    ("if (number_c_string[i] == '.') { number_c_string[i] = decimal_point; }", 0),
    ("if (number >= INT_MAX) { item->valueint = INT_MAX; }", 0),
    ("if (needed > INT_MAX) { return NULL; }", 0),
    ("if (p->noalloc) { return NULL; }", 0),
    ("if (needed <= INT_MAX) { newsize = INT_MAX; }", 0),
    ("if (i < 3) { h = h << 4; }", 0),
    ("if (*input_pointer != '\\\\') { *output_pointer++ = *input_pointer++; }", 0),
    ("if (output_buffer == NULL) { return false; }", 0),
    ("if (output == NULL) { return false; }", 0),
    ("if (*input_pointer < 32) { escape_characters += 5; }", 0),
    ("for (; 10 <= regular_total; regular_total /= 10) width++;", 2),
]

# Pre-fix failure indices (isalpha() bug active):      [9,10,11,17,18,24]
# Post-fix failure indices (identifier regex fix active): [0,9,10,11,15,17,18,24]
