import importlib
import os


class EnvSettings:
    def __init__(self):
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        self.results_path = '{}/tracking_results/'.format(root_path)
        self.network_path = '{}/networks/'.format(root_path)
        self.result_plot_path = '{}/result_plots/'.format(root_path)
        
        self.uav123_path = ''
        self.uav20l_path = ''
        self.uav123_10fps_path = ''
        self.uavdt_path = ''
        self.dtb70_path = ''
        self.visdrone2019sot_path = ''
        self.uav2022resiam_path = ''


def create_default_local_file():
    comment = {'results_path': 'Where to store tracking results',
               'network_path': 'Where tracking networks are stored.'}

    path = os.path.join(os.path.dirname(__file__), 'local.py')
    with open(path, 'w') as f:
        settings = EnvSettings()

        f.write('from .environment import EnvSettings\n\n')
        f.write('def local_env_settings():\n')
        f.write('    settings = EnvSettings()\n\n')
        f.write('    # Set your local paths here.\n\n')

        for attr in dir(settings):
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            attr_val = getattr(settings, attr)
            if not attr.startswith('__') and not callable(attr_val):
                if comment_str is None:
                    f.write('    settings.{} = \'{}\'\n'.format(attr, attr_val))
                else:
                    f.write('    settings.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))
        f.write('\n    return settings\n\n')


def env_settings():
    env_module_name = 'libs.dataset_info.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_env_settings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        # Create a default file
        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))
