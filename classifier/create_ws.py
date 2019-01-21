from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException


class AZHelper:
    def __init__(self):
        self.subscription_id = 'af7772d6-3275-4058-8d9e-7640742004e3'
        self.resource_group = 'Codefundopp'
        self.workspace_name = 'pyt1'

    @classmethod
    def load_ws(cls):
        return Workspace.from_config('/home/aaditya/PycharmProjects/Codefundopp/classifier/aml_config/config.json')

    @classmethod
    def load_cp(cls, ws):
        """
        Creates or loads a gpu cluster
        :param ws: Workspace
        :return:
        """
        compute_name = "gpucluster"
        compute_min_nodes = 0
        compute_max_nodes = 4

        # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
        # for cpu set SKU to STANDARD_D2_V2
        vm_size = "STANDARD_NC6"

        try:
            compute_target = ComputeTarget(workspace=ws, name=compute_name)
            print('found compute target. just use it. ' + compute_name)
        except ComputeTargetException:
            print('creating a new compute target...')
            provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                        min_nodes=compute_min_nodes,
                                                                        max_nodes=compute_max_nodes)

            # create the cluster
            compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
            # can poll for a minimum number of nodes and for a specific timeout.
            # if no min node count is provided it will use the scale settings for the cluster
            compute_target.wait_for_completion(show_output=True)

        # use get_status() to get a detailed status for the current AmlCompute.
        print(compute_target.get_status().serialize())
        return compute_target

    def create_ws(self):
        try:
            ws = Workspace.create(name=self.workspace_name,
                                  subscription_id=self.subscription_id,
                                  resource_group=self.resource_group,
                                  location='eastus2')
            ws.write_config()
            print('Library configuration succeeded')
        except:
            print('Workspace not found')
