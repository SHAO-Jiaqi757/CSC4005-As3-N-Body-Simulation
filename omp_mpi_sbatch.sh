#!/bin/bash
chmod +x ./build/*
tag="omp_mpi"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
SLURM_CPUS_PER_TASK=16
export LD_LIBRARY_PATH=./build/
rm -r tmp
mkdir -p tmp
for i in {1..8}
do
	for bodies in {20..500..20}
	do
	    line=$(squeue --me | wc -l)
	    while [ $line -gt 10 ]
	    do
		line=$(squeue --me | wc -l)
		echo "$line jobs in squeue"
		sleep 2s
	    done
	    echo "using $i cores"
	    echo "#!/bin/bash" > ./tmp/$i.sh
	    echo "export LD_LIBRARY_PATH=./build/" >> ./tmp/$i.sh
		echo "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}" >> ./tmp/$i.sh
	    echo "mpirun -n $i ./build/openmp_mpi $bodies 10 >> ./result/logs/${tag}-${dt}.log" >> ./tmp/$i.sh
	    cat ./tmp/$i.sh
	    sbatch --wait --account=csc4005 --partition=debug --qos=normal --nodes=4 --ntasks=$i --cpus-per-task=$SLURM_CPUS_PER_TASK ./tmp/$i.sh 
	done
done
