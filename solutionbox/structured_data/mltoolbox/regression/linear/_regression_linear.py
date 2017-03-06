from mltoolbox._structured_data import train_async as core_train

def train(train_dataset,
          eval_dataset,
          analysis_dir,
          output_dir,
          features,
          max_steps=5000,
          num_epochs=None,
          train_batch_size=100,
          eval_batch_size=16,
          min_eval_frequency=100,
          learning_rate=0.01,
          epsilon=0.0005,
          job_name=None, 
          cloud=None, 
          ):
  """Blocking version of train_async. See documentation for train_async."""
  job = train_async(
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      analysis_dir=analysis_dir,
      output_dir=output_dir,
      features=features,
      max_steps=max_steps,
      num_epochs=num_epochs,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      min_eval_frequency=min_eval_frequency,
      learning_rate=learning_rate,
      epsilon=epsilon,
      job_name=job_name, 
      cloud=cloud, 
      )
  job.wait()


def train(train_dataset,
          eval_dataset,
          analysis_dir,
          output_dir,
          features,
          max_steps=5000,
          num_epochs=None,
          train_batch_size=100,
          eval_batch_size=16,
          min_eval_frequency=100,
          learning_rate=0.01,
          epsilon=0.0005,
          job_name=None, 
          cloud=None, 
          ):
  """Train model locally or in the cloud.

  Args for local training:
    train_dataset: CsvDataSet
    eval_dataset: CsvDataSet
    analysis_dir:  The output directory from local_analysis
    output_dir:  Output directory of training.
    features: file path or features object. Example:
        {
          "col_A": {"transform": "scale", "default": 0.0},
          "col_B": {"transform": "scale","value": 4},
          # Note col_C is missing, so default transform used.
          "col_D": {"transform": "hash_one_hot", "hash_bucket_size": 4},
          "col_target": {"transform": "target"},
          "col_key": {"transform": "key"}
        }
        The keys correspond to the columns in the input files as defined by the
        schema file during preprocessing. Some notes
        1) The "key" and "target" transforms are required.
        2) Default values are optional. These are used if the input data has
           missing values during training and prediction. If not supplied for a
           column, the default value for a numerical column is that column's
           mean vlaue, and for a categorical column the empty string is used.
        3) For numerical colums, the following transforms are supported:
           i) {"transform": "identity"}: does nothing to the number. (default)
           ii) {"transform": "scale"}: scales the colum values to -1, 1.
           iii) {"transform": "scale", "value": a}: scales the colum values
              to -a, a.

           For categorical colums, the following transforms are supported:
          i) {"transform": "one_hot"}: A one-hot vector using the full
              vocabulary is used. (default)
          ii) {"transform": "embedding", "embedding_dim": d}: Each label is
              embedded into an d-dimensional space.
    max_steps: Int. Number of training steps to perform.
    num_epochs: Maximum number of training data epochs on which to train.
        The training job will run for max_steps or num_epochs, whichever occurs
        first.
    train_batch_size: number of rows to train on in one step.
    eval_batch_size: number of rows to eval in one step. One pass of the eval
        dataset is done. If eval_batch_size does not perfectly divide the numer
        of eval instances, the last fractional batch is not used.
    min_eval_frequency: Minimum number of training steps between evaluations.
    learning_rate: tf.train.AdamOptimizer's learning rate,
    epsilon: tf.train.AdamOptimizer's epsilon value.

  Args for cloud training:
    All local training arguments are valid for cloud training. Cloud training
    contains two additional args:

    cloud: A CloudTrainingConfig object.
    job_name: Training job name. A default will be picked if None.    
  Returns:
    Datalab job
  """
  return core_train(
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      analysis_dir=analysis_dir,
      output_dir=output_dir,
      features=features,
      model_type='linear_regression', 
      max_steps=max_steps,
      num_epochs=num_epochs,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      min_eval_frequency=min_eval_frequency,
      top_n=None,
      layer_sizes=None,
      learning_rate=learning_rate,
      epsilon=epsilon,
      job_name=job_name,
      job_name_prefix='mltoolbox_regression_linear',
      cloud=cloud,      
  )
