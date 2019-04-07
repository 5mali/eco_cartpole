#!/bin/bash

seedlist=(280 6738 4323 9295 1973 334 9213 7285 3121 394 5987 9588 9086 7544 862 3463 228 5253 6529 4821 7561 1466 3172 9826 2370 741 2493 6654 7551 6118 7628 3615 9948 7957 6192 1259 2712 9740 6040 1896 9684 5584 3531 371 5731 7100 4507 9382 4068 3713 2758 5632 7774 3644 1145 2429 7982 3854 6034 3217 6751 8367 8120 173 4850 297 9599 5290 7174 610 5955 5095 3119 3664 3497 2517 9836 1282 6325 3470 3520 985 8137 8079 5578 3857 7726 1858 5868 9247 427 4868 8954 1768 3099 2121 5418 5214 7873)

for seed in "${seedlist[@]}"
do
	rm ./Q_NPY/*.npy
    rm ./models/*.pt
    python ./ECO_A.py $seed >> ECO_A.out
done
