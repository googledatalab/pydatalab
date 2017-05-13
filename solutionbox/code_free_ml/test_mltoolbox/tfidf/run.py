#! /bin/bash

python ../../mltoolbox/code_free_ml/data/analyze_data.py --output-dir aout \
                       --csv-file-pattern ./train.csv \
                       --csv-schema-file schema.json \
                       --features-file features.json \
                       

python ../../mltoolbox/code_free_ml/data/transform_raw_data.py \
                             --csv-file-pattern ./train.csv \
                             --analyze-output-dir aout \
                             --output-filename-prefix ftrain \
                             --output-dir exout  \
                             --target \

python ../../mltoolbox/code_free_ml/data/transform_raw_data.py \
                             --csv-file-pattern ./eval.csv \
                             --analyze-output-dir aout \
                             --output-filename-prefix feval \
                             --output-dir exout  \
                             --target \


python dnnmodel.py \
--train-data-paths=./exout/ftrain* \
--eval-data-paths=./exout/feval* \
--job-dir tout \
--analysis-output-dir aout \
--model-type dnn_classification \
--layer-size1 10 \
--layer-size2 5 \
--top-n  3 \
--max-steps 500 

               