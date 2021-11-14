#!/bin/bash
chmod +x ./build/*
tag="as2-mpi"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
export LD_LIBRARY_PATH=./build/
rm -r tmp
mkdir -p tmp
for i in {1..1}
do
	for bodies in {20..20}
	do
	    line=$(squeue --me | wc -l)
	    while [ $line -gt 2 ]
	    do
		line=$(squeue --me | wc -l)
		echo "$line jobs in squeue"
		sleep 2s
	    done
	    echo "using $i cores"
	    echo "#!/bin/bash" > ./tmp/$i.sh
	    echo "export LD_LIBRARY_PATH=./build/" >> ./tmp/$i.sh
	    echo "mpirun -n $i ./build/mpi $bodies 1000 >> ./logs/${tag}-${dt}.log" >> ./tmp/$i.sh
	    cat ./tmp/$i.sh
	    sbatch --account=csc4005 --partition=debug --qos=normal --nodes=4 --ntasks=$i --distribution=cyclic ./tmp/$i.sh 
	done
done
