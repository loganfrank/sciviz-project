#!/bin/bash

conda activate research

nums=(1 2 3 4 5 6 7 8 9)
size=256

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}

    input_path="/home/logan/Desktop/code/sciviz-project/data/earth/data/nc/"
    output_path="/home/logan/Desktop/code/sciviz-project/data/earth/data/raw/"
    example="spherical00$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

nums=(0 1 2 3 4 5 6 7 8 9)

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}

    input_path="/home/logan/Desktop/code/sciviz-project/data/earth/data/nc/"
    output_path="/home/logan/Desktop/code/sciviz-project/data/earth/data/raw/"
    example="spherical01$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}

    input_path="/home/logan/Desktop/code/sciviz-project/data/earth/data/nc/"
    output_path="/home/logan/Desktop/code/sciviz-project/data/earth/data/raw/"
    example="spherical02$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}

    input_path="/home/logan/Desktop/code/sciviz-project/data/earth/data/nc/"
    output_path="/home/logan/Desktop/code/sciviz-project/data/earth/data/raw/"
    example="spherical03$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done