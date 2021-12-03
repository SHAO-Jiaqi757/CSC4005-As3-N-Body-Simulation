#!/bin/bash
chmod +x ./build/*
tag="omp_mpi"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
SLURM_CPUS_PER_TASK=4
export LD_LIBRARY_PATH=./build/
rm -r tmp
mkdir -p tmp
for i in {8..32..4}
do
	for bodies in {100..1000..100}
	do
	    line=$(squeue --me | wc -l)
	    while [ $line -gt 10 ]
	    do
		line=$(squeue --me | wc -l)
		echo "$line jobs in squeue"
		sleep 2s
	    done
		outfile="omp+mpi${i}-${bodies}.sh"
	    echo "using $i cores"
	    echo "#!/bin/bash" > ./tmp/outfile
	    echo "export LD_LIBRARY_PATH=./build/" >> ./tmp/outfile
		echo "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}" >> ./tmp/outfile
	    echo "mpirun -n $i ./build/openmp_mpi $bodies 4 >> ./result/logs/${tag}-${dt}.log" >> ./tmp/outfile
	    cat ./tmp/outfile
	    sbatch --account=csc4005 --partition=debug --qos=normal --nodes=4 --ntasks=$i --cpus-per-task=$SLURM_CPUS_PER_TASK ./tmp/outfile
	done
done
