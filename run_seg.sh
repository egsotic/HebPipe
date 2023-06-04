server=$1
gpu_i=$2
input_path=$3
output_path=$4

cmd="cd /home/nlp/egsotic/repo/HebPipe && source /home/nlp/egsotic/repo/HebPipe/venv-hebpipe/bin/activate && env CUDA_VISIBLE_DEVICES="${gpu_i}" nohup python run_seg.py --input_path "${input_path}" --output_path "${output_path}" > /dev/null 2>&1 &"
ssh ${server} "${cmd}"