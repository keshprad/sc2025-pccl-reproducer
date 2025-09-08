# NOTE: First set library to either "pccl" or "xccl" in run_raw_collectives_benchmark.sh
sbatch -N 8 -n 32 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 8 -n 32 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 8 -n 32 scripts/perlmutter/run_raw_collectives_benchmark.sh


sbatch -N 16 -n 64 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 16 -n 64 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 16 -n 64 scripts/perlmutter/run_raw_collectives_benchmark.sh


sbatch -N 32 -n 128 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 32 -n 128 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 32 -n 128 scripts/perlmutter/run_raw_collectives_benchmark.sh


sbatch -N 64 -n 256 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 64 -n 256 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 64 -n 256 scripts/perlmutter/run_raw_collectives_benchmark.sh


sbatch -N 128 -n 512 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 128 -n 512 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 128 -n 512 scripts/perlmutter/run_raw_collectives_benchmark.sh


sbatch -N 256 -n 1024 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 256 -n 1024 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 256 -n 1024 scripts/perlmutter/run_raw_collectives_benchmark.sh


sbatch -N 512 -n 2048 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 512 -n 2048 scripts/perlmutter/run_raw_collectives_benchmark.sh
sbatch -N 512 -n 2048 scripts/perlmutter/run_raw_collectives_benchmark.sh
