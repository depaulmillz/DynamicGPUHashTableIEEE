#!/bin/bash

#This is a test script to collect data

size=( 10000 100000 500000 1000000 )
write=( 0 5 10 15 20 )

for p in "${write[@]}"
do
    for i in "${size[@]}"
    do
        timeout 15m numactl ./OnlineThroughputTest -m $i -w $p > "$p-$i.txt" 2>&1
    done
done
