export TF_CPP_MIN_LOG_LEVEL=2
para_dir=parameter
mkdir -p $para_dir

# python -u src/run.py --train --model SA --data_dir data --w2v_file data/glove.840B.300d.txt --save_dir $para_dir
# python -u src/run.py --test --model KA+D --data_dir data --w2v_file data/glove.840B.300d.txt --save_dir $para_dir --load_model paper_parameter/ka+d/model
python -u src/run.py --direct --model KA+D --data_dir data --direct_dir direct --w2v_file data/glove.840B.300d.txt --save_dir $para_dir --load_model paper_parameter/ka+d/model
