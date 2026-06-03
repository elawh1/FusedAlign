#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --job-name=job_ER_avg_June2_2
#SBATCH --output=output/job_ER_avg_June2_2.out
#SBATCH --error=error/job_ER_avg_June2_2.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

module load PrgEnv-intel

export MKL_NUM_THREADS=48 OMP_NUM_THREADS=48 MKL_DYNAMIC=FALSE

DATA_PATH="/global/homes/d/dmandara/network_alignment/Parrot-Imp/ER-Graph/ER-Datasets"
tmpdir="${PARROT_TMPDIR:-/global/homes/d/dmandara/network_alignment/Parrot-Imp/Parrot/scripts/tmp}"
mkdir -p "$tmpdir"
script_tag="${SLURM_JOB_NAME:-$(basename "$0" .sh)}_${SLURM_JOB_ID:-$$}"

for DATA in "g40k" "g50k" "g60k" "g70k" "g80k" "g90k"; do
	cmd="./../parrot $DATA_PATH/$DATA 0"   # replace with your command
	runs=11
	run_prefix="$tmpdir/${script_tag}_${DATA}"
	trap 'rm -f "$run_prefix"_run_*.txt' EXIT

	echo "------ STARTING RUNS FOR $DATA ------"

	for i in $(seq -w 1 $runs); do
	echo "RUN $i/$runs"
	$cmd > "${run_prefix}_run_$i.txt"
	done

	awk '
	/^total phases of algorithm completed/ { exit }
	{ print }
	' "${run_prefix}_run_01.txt"

	echo
	echo "[AVERAGE] {"

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
	' "${run_prefix}"_run_0[2-9].txt "${run_prefix}"_run_1[0-1].txt
	echo "}"
	echo "------ END RUNS FOR $DATA ------"
	rm -f "${run_prefix}"_run_*.txt
	trap - EXIT
done
