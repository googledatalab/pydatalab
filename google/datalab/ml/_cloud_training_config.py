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

from collections import namedtuple

_CloudTrainingConfig = namedtuple("CloudConfig",
                                  ['region', 'scale_tier', 'master_type', 'worker_type',
                                   'parameter_server_type', 'worker_count',
                                   'parameter_server_count'])
_CloudTrainingConfig.__new__.__defaults__ = ('BASIC', None, None, None, None, None)


class CloudTrainingConfig(_CloudTrainingConfig):
    """A config namedtuple containing cloud specific configurations for CloudML training.

    Fields:
      region: the region of the training job to be submitted. For example, "us-central1".
          Run "gcloud compute regions list" to get a list of regions.
      scale_tier: Specifies the machine types, the number of replicas for workers and
          parameter servers. For example, "STANDARD_1". See
          https://cloud.google.com/ml/reference/rest/v1beta1/projects.jobs#scaletier
          for list of accepted values.
      master_type: specifies the type of virtual machine to use for your training
          job's master worker. Must set this value when scale_tier is set to CUSTOM.
          See the link in "scale_tier".
      worker_type: specifies the type of virtual machine to use for your training
          job's worker nodes. Must set this value when scale_tier is set to CUSTOM.
      parameter_server_type: specifies the type of virtual machine to use for your training
          job's parameter server. Must set this value when scale_tier is set to CUSTOM.
      worker_count: the number of worker replicas to use for the training job. Each
          replica in the cluster will be of the type specified in "worker_type".
          Must set this value when scale_tier is set to CUSTOM.
      parameter_server_count: the number of parameter server replicas to use. Each
          replica in the cluster will be of the type specified in "parameter_server_type".
          Must set this value when scale_tier is set to CUSTOM.
    """
    pass
