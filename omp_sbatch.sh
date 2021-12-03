#!/bin/bash
chmod +x ./build/*
tag="omp"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
export LD_LIBRARY_PATH=./build/
mkdir -p tmp
for thread_num in 1 5 9 16 24 30 32 33 35
do
    for bodies in {600..1000..100}
    do
        line=$(squeue --me | wc -l)
        while [ $line -gt 10 ]
        do
        line=$(squeue --me | wc -l)
        echo "$line jobs in squeue"
        sleep 2s
        done
        echo "#!/bin/bash" > ./tmp/$bodies.sh
        echo "export LD_LIBRARY_PATH=./build/" >> ./tmp/$bodies.sh
        echo "./build/openmp $thread_num $bodies 4 >> ./result/logs/${tag}-${dt}.log" >> ./tmp/$bodies.sh

        cat ./tmp/$bodies.sh
        sbatch --wait --account=csc4005 --partition=debug --qos=normal  --nodes=1 --ntasks-per-node=32 --ntasks=32 ./tmp/$bodies.sh
    done
done
