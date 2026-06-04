#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parrot_dir="$(cd "$script_dir/.." && pwd)"

export MKL_NUM_THREADS="${MKL_NUM_THREADS:-48}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-48}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

DATA_PATH="${PARROT_DATA_PATH:-$HOME/cdir-m4293/durga/parrot-data1}"
parrot_bin="${PARROT_BIN:-$parrot_dir/parrot}"
runs="${PARROT_RUNS:-11}"
tmpdir="${PARROT_TMPDIR:-$script_dir/tmp}"
mkdir -p "$tmpdir"
script_tag="$(basename "$0" .sh)"
output_dir="${PARROT_OUTPUT_DIR:-$script_dir/output}"
error_dir="${PARROT_ERROR_DIR:-$script_dir/error}"
mkdir -p "$output_dir" "$error_dir"
exec > "$output_dir/${script_tag}.out"
exec 2> "$error_dir/${script_tag}.err"

if [[ ! -x "$parrot_bin" ]]; then
	echo "parrot not found or not executable: $parrot_bin" >&2
	exit 1
fi
if (( runs < 2 )); then
	echo "PARROT_RUNS must be at least 2 for averaging" >&2
	exit 1
fi

datasets="${PARROT_DATASETS:-GGI acm-dbpl arxiv dbp15k flicker-lastfm twitter-foursquare}"

for DATA in $datasets ; do
	run_prefix="$tmpdir/${script_tag}_${DATA}"
	trap 'rm -f "$run_prefix"_run_*.txt' EXIT

	echo "------ STARTING RUNS FOR $DATA ------"

	for i in $(seq -w 1 $runs); do
	echo "RUN $i/$runs"
	"$parrot_bin" "$DATA_PATH/$DATA" 0 > "${run_prefix}_run_$i.txt"
	done

	awk '
	/^total phases of algorithm completed/ { exit }
	{ print }
	' "${run_prefix}_run_01.txt"

	echo
	echo "[AVERAGE] {"

	avg_files=()
	for i in $(seq -w 2 "$runs"); do
		avg_files+=("${run_prefix}_run_$i.txt")
	done

	awk '
	function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
	function fmt(x) { s = sprintf("%.6f", x); sub(/\.?0+$/, "", s); return s }
	function parse_line(line,   prefix, num, suffix) {
		if (match(line, /([0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?)([[:space:]]*ms)?[[:space:]]*$/, m)) {
		prefix = substr(line, 1, RSTART-1)
		num = m[1]
		suffix = substr(line, RSTART + RLENGTH)
		return (key = trim(prefix)) && (value = num) && (pref = prefix) && (suf = suffix)
		}
		return 0
	}
	BEGIN { section = 0; first = ARGV[1]; blank_count = 0 }
	{
		if (!section && /^total phases of algorithm completed/) section = 1
		if (section) {
		if (/^[[:space:]]*$/) {
			blank_count++
			if (blank_count > 1) section = 0
			else { line_type[++n] = "text"; line_text[n] = $0 }
		} else {
			blank_count = 0
			if (FILENAME == first) {
			if (parse_line($0)) {
				line_type[++n] = "metric"
				line_key[n] = key
				line_prefix[n] = pref
				line_suffix[n] = suf
			} else {
				line_type[++n] = "text"
				line_text[n] = $0
			}
			}
			if (FILENAME != first && parse_line($0)) {
			sum[key] += value
			cnt[key]++
			}
		}
		}
	}
	END {
		for (i = 1; i <= n; i++) {
		if (line_type[i] == "text") {
			print line_text[i]
		} else {
			key = line_key[i]
			printf "%s%s%s\n", line_prefix[i], fmt(sum[key] / cnt[key]), line_suffix[i]
		}
		}
	}
	' "${avg_files[@]}"
	echo "}"
	echo "------ END RUNS FOR $DATA ------"
	rm -f "${run_prefix}"_run_*.txt
	trap - EXIT
done
