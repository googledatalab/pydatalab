# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
import random
import json
import subprocess


def make_csv_data(filename, num_rows, problem_type, keep_target=True):
  """Writes csv data for preprocessing and training.

  Args:
    filename: writes data to csv file.
    num_rows: how many rows of data will be generated.
    problem_type: 'classification' or 'regression'. Changes the target value.
    keep_target: if false, the csv file will have an empty column ',,' for the 
        target.
  """
  random.seed(12321)
  with open(filename, 'w') as f1:
    for i in range(num_rows):
      num1 = random.uniform(0, 30)
      num2 = random.randint(0, 20)
      num3 = random.uniform(0, 10)

      str1 = random.choice(['red', 'blue', 'green', 'pink', 'yellow', 'brown', 'black'])
      str2 = random.choice(['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr'])
      str3 = random.choice(['car', 'truck', 'van', 'bike', 'train', 'drone'])

      map1 = {'red': 2, 'blue': 6, 'green': 4, 'pink': -5, 'yellow': -6, 'brown': -1, 'black': 7}
      map2 = {'abc': 10, 'def': 1, 'ghi': 1, 'jkl': 1, 'mno': 1, 'pqr': 1}
      map3 = {'car': 5, 'truck': 10, 'van': 15, 'bike': 20, 'train': 25, 'drone': 30}

      # Build some model.
      t = 0.5 + 0.5*num1 -2.5*num2 + num3
      t += map1[str1] + map2[str2] + map3[str3]

      if problem_type == 'classification':
        if t < 0:
          t = 100
        elif t < 20:
          t = 101
        else:
          t = 102

      if keep_target:
          csv_line = "{id},{target},{num1},{num2},{num3},{str1},{str2},{str3}\n".format(
            id=i,
            target=t,
            num1=num1,
            num2=num2,
            num3=num3,
            str1=str1,
            str2=str2,
            str3=str3)
      else:
          csv_line = "{id},{num1},{num2},{num3},{str1},{str2},{str3}\n".format(
            id=i,
            num1=num1,
            num2=num2,
            num3=num3,
            str1=str1,
            str2=str2,
            str3=str3)
      f1.write(csv_line)


def make_preprocess_schema(filename, problem_type):
  """Makes a schema file compatable with the output of make_csv_data.

  Writes a json file.

  Args:
    filename: output file path
    problem_type: regression or classification
  """
  schema = [
      {
          "mode": "NULLABLE",
          "name": "key",
          "type": "STRING"
      },
      {
          "mode": "REQUIRED",
          "name": "target",
          "type": ("STRING" if problem_type == 'classification' else "FLOAT")
      },        
      {
          "mode": "NULLABLE",
          "name": "num1",
          "type": "FLOAT"
      },
      {
          "mode": "NULLABLE",
          "name": "num2",
          "type": "INTEGER"
      },
      {
          "mode": "NULLABLE",
          "name": "num3",
          "type": "FLOAT"
      },
      {
          "mode": "NULLABLE",
          "name": "str1",
          "type": "STRING"
      },
      {
          "mode": "NULLABLE",
          "name": "str2",
          "type": "STRING"
      },
      {
          "mode": "NULLABLE",
          "name": "str3",
          "type": "STRING"
      }  
  ]
  with open(filename, 'w') as f:
    f.write(json.dumps(schema))


def run_preprocess(output_dir, csv_filename, schema_filename):
  preprocess_script = os.path.abspath(
      os.path.join(os.path.dirname(__file__), 
                   '../preprocess/local_preprocess.py'))

  cmd = ['python', preprocess_script,
         '--output_dir', output_dir,
         '--input_file_pattern', csv_filename,
         '--schema_file', schema_filename
  ]
  print('Going to run command: %s' % ' '.join(cmd))
  subprocess.check_call(cmd) #, stderr=open(os.devnull, 'wb'))


def run_training(
      train_data_paths,
      eval_data_paths,
      output_path,
      preprocess_output_dir,
      transforms_file,
      max_steps,
      model_type,
      extra_args=[]):
  """Runs Training via gcloud beta ml local train.

  Args:
    train_data_paths: training csv files
    eval_data_paths: eval csv files
    output_path: folder to write output to
    preprocess_output_dir: output location of preprocessing
    transforms_file: path to transforms file
    max_steps: max training steps
    model_type: {dnn,linear}_{regression,classification}
    extra_args: array of strings, passed to the trainer.

  Returns:
    The stderr of training as one string. TF writes to stderr, so basically, the
    output of training.
  """

  # Gcloud has the fun bug that you have to be in the parent folder of task.py
  # when you call it. So cd there first.
  task_parent_folder = os.path.abspath(
      os.path.join(os.path.dirname(__file__), '..'))
  cmd = ['cd %s &&' % task_parent_folder,
         'gcloud beta ml local train',
         '--module-name=trainer.task',
         '--package-path=trainer',
         '--',
         '--train_data_paths=%s' % train_data_paths,
         '--eval_data_paths=%s' % eval_data_paths,
         '--output_path=%s' % output_path,
         '--preprocess_output_dir=%s' % preprocess_output_dir,
         '--transforms_file=%s' % transforms_file,
         '--model_type=%s' % model_type,
         '--max_steps=%s' % max_steps] + extra_args
  print('Going to run command: %s' % ' '.join(cmd))
  sp = subprocess.Popen(' '.join(cmd), shell=True, stderr=subprocess.PIPE)
  _, err = sp.communicate()
  return err

if __name__ == '__main__':
  make_csv_data('raw_train_regression.csv', 5000, 'regression', True)
  make_csv_data('raw_eval_regression.csv', 1000, 'regression', True)
  make_csv_data('raw_predict_regression.csv', 100, 'regression', False)
  make_preprocess_schema('schema_regression.json', 'regression')

  make_csv_data('raw_train_classification.csv', 5000, 'classification', True)
  make_csv_data('raw_eval_classification.csv', 1000, 'classification', True)
  make_csv_data('raw_predict_classification.csv', 100, 'classification', False)
  make_preprocess_schema('schema_classification.json', 'classification')

