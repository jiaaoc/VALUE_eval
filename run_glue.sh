:<<!
task = 'MNLI'
lr = 5e-6
postfix = 'roberta_base_VALUE'
project = 'prefix_mnli'

script_params = {
    "--model_name_or_path": "roberta-base",
    "--data_dir": datastore.path(train_file+'VALUE/' + task).as_mount(),
    "--task_name": task,
    "--do_train":None,
    "--do_eval":None,
    "--evaluate_during_training":None,
    "--learning_rate":lr,
    "--num_train_epochs":5.0,
    "--logging_steps":500,
    "--save_steps":500,
    "--per_gpu_train_batch_size":16,
    "--output_dir":datastore.path(output_dir+task+"-"+postfix).as_mount(),
    "--overwrite_output_dir":None,
    "--seed": 42,
}
custom_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"


mpi = MpiConfiguration()
mpi.process_count_per_node = 1

estimator = PyTorch(
    compute_target=compute_target,
    use_gpu=True,
    use_docker=True,
    node_count=1,
    distributed_training=mpi,
    custom_docker_image=custom_image,
    pip_packages=["setuptools", "transformers", "tensorboardX", "numpy", "scikit-learn",
                  "boto3", "requests", "tqdm", "regex", "sentencepiece", "sacremoses"],
    source_directory=source_directory,
    entry_script=entry_script,
    script_params=script_params
)
experiment = Experiment(ws, name=project)
run = experiment.submit(estimator)
run.wait_for_completion(show_output=False)
!




export GLUE_DIR=./glue_data
export TASK_NAME=SST-2

CUDA_VISIBLE_DEVICES=0 \
python run_glue.py \
  --model_name_or_path roberta-base \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_aug \
  --aug_type 'span_cutoff' \
  --aug_cutoff_ratio 0.1 \
  --aug_ce_loss 1.0 \
  --aug_js_loss 1.0 \
  --learning_rate 5e-6 \
  --num_train_epochs 10.0 \
  --logging_steps 500 \
  --save_steps 500 \
  --per_gpu_train_batch_size 8 \
  --output_dir results/$TASK_NAME-roberta_base-cutoff \
  --overwrite_output_dir
