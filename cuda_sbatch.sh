#!/bin/bash
chmod +x ./build/*
tag="cuda"
dt=$(date '+%d-%m-%Y-%H:%M:%S')
export LD_LIBRARY_PATH=./build/
mkdir -p tmp
for thread_num in {10..100..5}
do
    for bodies in {200..1000..80}
    do
        line=$(squeue --me | wc -l)
        while [ $line -gt 10 ]
        do
        line=$(squeue --me | wc -l)
        echo "$line jobs in squeue"
        sleep 2s
        done
        outfile="cuda${thread_num}-${bodies}.sh"
        echo "#!/bin/bash" > ./tmp/outfile
        echo "export LD_LIBRARY_PATH=./build/" >> ./tmp/outfile
        echo "./build/cuda $thread_num $bodies 10 >> ./result/logs/${tag}-${dt}.log" >> ./tmp/outfile

        cat ./tmp/outfile
        sbatch --wait --account=csc4005 --partition=debug --qos=normal --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 ./tmp/outfile
    done
done
