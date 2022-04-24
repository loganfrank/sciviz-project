#!/bin/bash

nums=(1 2 3 4 5 6 7 8 9)
size=128

input_path="/Users/loganfrank/Desktop/code/sciviz-project/data/earth/data/nc/"
output_path="/Users/loganfrank/Desktop/code/sciviz-project/data/earth/data/raw/"

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical00$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

nums=(0 1 2 3 4 5 6 7 8 9)

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical01$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical02$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical03$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical04$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical05$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical06$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical07$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical08$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical09$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical10$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical11$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical12$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical13$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical14$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

for (( i=0; i<${#nums[@]} ; i+=1 )) ; do
    num=${nums[i]}
    example="spherical15$num"
    python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
done

example="spherical160"
python convert_earth_data.py --input_path $input_path --output_path $output_path --example $example --size $size
