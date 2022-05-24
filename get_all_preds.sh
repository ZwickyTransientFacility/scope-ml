#!/bin/bash

for ccd in {1..16}
do
    for quad in {1..4}
    do
        # echo -n "agn ... " && python -m profile -s time tools/inference.py --path-model=models/dr5-1/agn-20210919_090902.h5  --model-class=agn --ccd=1 --quad=1 --flag_ids && echo "done"
        echo "ccd "$ccd" quad "$quad" ... "
        echo -n "agn ... "  && python tools/inference.py --path-model=models/dr5-1/agn-20210919_090902.h5   --model-class=agn --ccd=$ccd --quad=$quad --flag_ids && echo "done"
        echo -n "bis ... "  && python tools/inference.py --path-model=models/dr5-1/bis-20210919_140037.h5   --model-class=bis --ccd=$ccd --quad=$quad && echo "done"
        echo -n "blyr ... " && python tools/inference.py --path-model=models/dr5-1/blyr-20210919_184548.h5  --model-class=blyr --ccd=$ccd --quad=$quad && echo "done"
        echo -n "ceph ... " && python tools/inference.py --path-model=models/dr5-1/ceph-20210919_190517.h5  --model-class=ceph --ccd=$ccd --quad=$quad && echo "done"
        echo -n "dscu ... " && python tools/inference.py --path-model=models/dr5-1/dscu-20210919_204537.h5  --model-class=dscu --ccd=$ccd --quad=$quad && echo "done"
        echo -n "e ... "    && python tools/inference.py --path-model=models/dr5-1/e-20210918_225015.h5     --model-class=e --ccd=$ccd --quad=$quad && echo "done"
        echo -n "ea ... "   && python tools/inference.py --path-model=models/dr5-1/ea-20210918_222613.h5    --model-class=ea --ccd=$ccd --quad=$quad && echo "done"
        echo -n "eb ... "   && python tools/inference.py --path-model=models/dr5-1/eb-20210918_214407.h5    --model-class=eb --ccd=$ccd --quad=$quad && echo "done"
        echo -n "ew ... "   && python tools/inference.py --path-model=models/dr5-1/ew-20210918_144924.h5    --model-class=ew --ccd=$ccd --quad=$quad && echo "done"
        echo -n "fla ... "  && python tools/inference.py --path-model=models/dr5-1/fla-20210918_080316.h5   --model-class=fla --ccd=$ccd --quad=$quad && echo "done"
        echo -n "i ... "    && python tools/inference.py --path-model=models/dr5-1/i-20210918_071727.h5     --model-class=i --ccd=$ccd --quad=$quad && echo "done"
        echo -n "longt ... " && python tools/inference.py --path-model=models/dr5-1/longt-20210918_051118.h5 --model-class=longt --ccd=$ccd --quad=$quad && echo "done"
        echo -n "lpv ... "  && python tools/inference.py --path-model=models/dr5-1/lpv-20210919_213033.h5   --model-class=lpv --ccd=$ccd --quad=$quad && echo "done"
        echo -n "pnp ... "  && python tools/inference.py --path-model=models/dr5-1/pnp-20210918_041303.h5   --model-class=pnp --ccd=$ccd --quad=$quad && echo "done"
        echo -n "puls ... " && python tools/inference.py --path-model=models/dr5-1/puls-20210920_012555.h5  --model-class=puls --ccd=$ccd --quad=$quad && echo "done"
        echo -n "rrlyr ... " && python tools/inference.py --path-model=models/dr5-1/rrlyr-20210920_041205.h5 --model-class=rrlyr --ccd=$ccd --quad=$quad && echo "done"
        echo -n "rscvn ... " && python tools/inference.py --path-model=models/dr5-1/rscvn-20210920_045003.h5 --model-class=rscvn --ccd=$ccd --quad=$quad && echo "done"
        echo -n "srv ... "  && python tools/inference.py --path-model=models/dr5-1/srv-20210920_053008.h5   --model-class=srv --ccd=$ccd --quad=$quad && echo "done"
        echo -n "vnv ... "  && python tools/inference.py --path-model=models/dr5-1/vnv-20210915_220725.h5   --model-class=vnv --ccd=$ccd --quad=$quad && echo "done"
        echo -n "yso ... "  && python tools/inference.py --path-model=models/dr5-1/yso-20210920_184534.h5   --model-class=yso --ccd=$ccd --quad=$quad && echo "done"
        echo -n "combining ... " && python combine_preds.py --ccd=$ccd --quad=$quad && echo "done"
        echo "completed inference for ccd "$ccd" quad "$quad
    done
done
