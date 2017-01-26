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
import subprocess


def make_csv_data(filename, num_rows, problem_type):
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
          t = 1
        elif t < 20:
          t = 2
        else:
          t = 3

      csv_line = "{id},{target},{num1},{num2},{num3},{str1},{str2},{str3}\n".format(
          id=i,
          target=t,
          num1=num1,
          num2=num2,
          num3=num3,
          str1=str1,
          str2=str2,
          str3=str3)
      f1.write(csv_line)

  schema = {'column_names': ['key', 'target', 'num1', 'num2', 'num3',
                             'str1', 'str2', 'str3'],
            'key_column': 'key',
            'target_column': 'target',
            'numerical_columns': ['num1', 'num2', 'num3'],
            'categorical_columns': ['str1', 'str2', 'str3']
  }
  if problem_type == 'classification':
    schema['categorical_columns'] += ['target']
  else:
    schema['numerical_columns'] += ['target']

  # use defaults for num3 and str3
  transforms = {'num1': {'transform': 'identity'},
                'num2': {'transform': 'identity'},
  #              'num3': {'transform': 'identity'},
                'str1': {'transform': 'one_hot'},
                'str2': {'transform': 'one_hot'},
  #              'str3': {'transform': 'one_hot'}
  }
  return schema, transforms


def run_preprocess(output_dir, csv_filename, schema_filename,
                   train_percent='80', eval_percent='10', test_percent='10'):
  cmd = ['python', './preprocess/preprocess.py',
         '--output_dir', output_dir,
         '--input_file_path', csv_filename,
         '--schema_file', schema_filename,
         '--train_percent', train_percent,
         '--eval_percent', eval_percent,
         '--test_percent', test_percent,
  ]
  print('Current working directoyr: %s' % os.getcwd())
  print('Going to run command: %s' % ' '.join(cmd))
  subprocess.check_call(cmd, stderr=open(os.devnull, 'wb'))


def run_training(output_dir, input_dir, schema_filename, transforms_filename,
                 extra_args=[]):
  """Runs Training via gcloud beta ml local train.

  Args:
    output_dir: the trainer's output folder
    input_dir: should contain features_train*, features_eval*, and
        mmetadata.json.
    schema_filename: path to the schema file
    transforms_filename: path to the transforms file.
    extra_args: array of strings, passed to the trainer.
  """
  train_filename = os.path.join(input_dir, 'features_train*')
  eval_filename = os.path.join(input_dir, 'features_eval*')
  metadata_filename = os.path.join(input_dir, 'metadata.json')

  cmd = ['gcloud beta ml local train',
         '--module-name=trainer.task',
         '--package-path=trainer',
         '--',
         '--train_data_paths=%s' % train_filename,
         '--eval_data_paths=%s' % eval_filename,
         '--metadata_path=%s' % metadata_filename,
         '--output_path=%s' % output_dir,
         '--schema_file=%s' % schema_filename,
         '--transforms_file=%s' % transforms_filename,
         '--max_steps=2500'] + extra_args
  print('Current working directoyr: %s' % os.getcwd())
  print('Going to run command: %s' % ' '.join(cmd))
  # Don't print all the training output, just the last line that has loss value.
  sp = subprocess.Popen(' '.join(cmd), shell=True, stderr=subprocess.PIPE)
  _, err = sp.communicate()
  err = err.splitlines()
  print 'last line'
  print err[len(err)-1]
