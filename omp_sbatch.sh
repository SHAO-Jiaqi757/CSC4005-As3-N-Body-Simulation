#!/bin/bash
chmod +x ./build/*
tag="omp"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
export LD_LIBRARY_PATH=./build/
mkdir -p tmp
for thread_num in {1..32}
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
        echo "#!/bin/bash" > ./tmp/thread_num=$thread_num.sh
        echo "export LD_LIBRARY_PATH=./build/" >> ./tmp/thread_num=$thread_num.sh
        echo "./build/openmp $thread_num $bodies 1000 >> ./result/logs/${tag}-${dt}.log" >> ./tmp/thread_num=$thread_num.sh

        cat ./tmp/thread_num=$thread_num.sh
        sbatch --account=csc4005 --partition=debug --qos=normal  --nodes=1 --ntasks-per-node=32 --ntasks=32 ./tmp/thread_num=$thread_num.sh
    done
done
