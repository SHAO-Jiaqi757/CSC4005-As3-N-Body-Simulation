#!/bin/bash
chmod +x ./build/*
tag="mpi"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
export LD_LIBRARY_PATH=./build/
rm -r tmp
mkdir -p tmp
for i in {1..128..8}
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
	    echo "#!/bin/bash" > ./tmp/$bodies.sh
	    echo "export LD_LIBRARY_PATH=./build/" >> ./tmp/$bodies.sh
	    echo "mpirun -n $i ./build/mpi $bodies 1000 >> ./result/logs/${tag}-${dt}.log" >> ./tmp/$bodies.sh
	    cat ./tmp/$bodies.sh
	    sbatch --account=csc4005 --partition=debug --qos=normal --nodes=4 --ntasks=$i ./tmp/$bodies.sh
	done
done
