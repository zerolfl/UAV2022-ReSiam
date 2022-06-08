from .environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.dtb70_path = 'D:/Documents/Datasets/DTB70'
    settings.network_path = 'd:/Documents/Codes/pytracking/networks/'    # Where tracking networks are stored.
    settings.result_plot_path = 'd:/Documents/Codes/pytracking/result_plots/'
    settings.results_path = 'd:/Documents/Codes/pytracking/tracking_results/'    # Where to store tracking results
    settings.uav123_10fps_path = 'D:/Documents/Datasets/UAV123_10fps'
    settings.uav123_path = 'D:/Documents/Datasets/UAV123'
    settings.uav20l_path = 'D:/Documents/Datasets/UAV123'
    settings.uav2022resiam_path = 'D:/Documents/Datasets/UAV2022-ReSiam'

    return settings

