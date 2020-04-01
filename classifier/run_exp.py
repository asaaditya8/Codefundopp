from azureml.core.runconfig import RunConfiguration
from classifier.create_ws import AZHelper
from azureml.core import Experiment, ScriptRunConfig

address = '13.68.140.115'
script_folder = 'azure'

ws = AZHelper.load_ws()
compute_target = AZHelper.load_vm(ws, address)

run_dsvm = RunConfiguration(framework='python')
run_dsvm.target = compute_target
run_dsvm.environment.python.user_managed_dependencies = True
run_dsvm.environment.python.interpreter_path = '/data/anaconda/envs/py36/bin/python'

experiment_name = 'pytorch-distr-hvd'
exp = Experiment(ws, name=experiment_name)

src = ScriptRunConfig(source_directory = script_folder, script = 'train.py', run_config = run_dsvm)
run = exp.submit(src)
run.wait_for_completion(show_output = True)

model = run.register_model(model_name='xception_wf_1',
                           model_path='./outputs/xception_wf_rvm_2.pth')
print(model.name, model.id, model.version, sep = '\t')