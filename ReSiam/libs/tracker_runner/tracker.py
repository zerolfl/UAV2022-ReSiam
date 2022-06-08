
import os
import sys
import time
import torch
import importlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from collections import OrderedDict

from libs.visualization.visdom import Visdom
from libs.dataset_info.environment import env_settings
from libs.visualization.plotting import draw_figure, overlay_mask
from .multi_object_wrapper import MultiObjectWrapper


_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}


def trackerlist(name: str, parameter_name: str, run_ids = None, display_name: str = None):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, run_id, display_name) for run_id in run_ids]  # 同1个tracker的可能要跑很多轮次取平均结果

class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, 
                 name: str, 
                 parameter_name: str, 
                 run_id: int = None, 
                 display_name: str = None, 
                 hyper_params: dict = None,
                 experiment_name: str = None,
                 all_hyper_params_dict: dict = None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name                      # 跟踪器名称
        self.parameter_name = parameter_name  # 跟踪器参数
        self.run_id = run_id                  # 跟踪器轮次id（多次进行同一个跟踪器的时候需要）
        self.display_name = display_name      # 展示结果时用的名称
        self.hyper_params = hyper_params      # 用于超参数搜索
        self.experiment_name = experiment_name  # 若有超参数, 则保存结果的路径增加实验名
        
        self.all_hyper_params_dict = all_hyper_params_dict  # 本次实验所有的超参数

        env = env_settings()                  # 读取设置的各类数据集路径以及结果保存路径
        if self.experiment_name is None:
            if self.run_id is None:
                self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            else:
                self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        else:
            if self.run_id is None:
                self.results_dir = '{}/{}-{}/{}'.format(env.results_path, self.name, self.experiment_name, self.parameter_name)
            else:
                self.results_dir = '{}/{}-{}/{}_{:03d}'.format(env.results_path, self.name, self.experiment_name, self.parameter_name, self.run_id)
                
        self.default_param_file = importlib.import_module('parameter.{}.{}'.format(self.name, self.parameter_name)).__file__
        
        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tracker', self.name))  # 得到跟踪器模块路径
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module('tracker.{}'.format(self.name))  # 导入对应模块
            self.tracker_class = tracker_module.get_tracker_class()                              # 读取该模块下所指的跟踪器类
        else:
            self.tracker_class = None

        self.visdom = None
        self.pause_mode = False
        self.step = False
        self.show_search_region = False
        self.show_response_region = False

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        # self.pause_mode = False
        # self.step = False
        # self.show_search_region = False
        # self.show_response_region = False
        
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                # self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                #                      visdom_info=visdom_info)
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': '跟踪过程'},
                                     visdom_info=visdom_info)
                
                # Show help
                # help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                #             'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                #             'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                #             'block list.'
                # self.visdom.register(help_text, 'text', 1, 'Help')
                
                cn_help_text = '<img src=\"https://s1.ax1x.com/2022/05/01/O9tQIg.gif\", width=\"20\"><b>功能键</b><br>' \
                               '<img src=\"https://s1.ax1x.com/2022/05/01/O9tIWd.png\", width=\"28\">空格: 进入/退出暂停模式<br>' \
                               '<img src=\"https://s1.ax1x.com/2022/05/01/O9N0nP.png\", width=\"28\">方向键: 暂停模式下跟踪下一帧<br>' \
                               '<img src=\"https://s1.ax1x.com/2022/05/01/O9N57T.png\", width=\"28\">: 显示特征提取区域<br>' \
                               '<img src=\"https://s1.ax1x.com/2022/05/13/Ordy26.png\", width=\"28\">: 显示响应计算区域<br>'
                self.visdom.register(cn_help_text, 'text', 1, '帮助')
                               
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True
                
            if data['key'] == 's':
                self.show_search_region = not self.show_search_region
            if data['key'] == 'x':
                self.show_response_region = not self.show_response_region


    def create_tracker(self, params):
        tracker = self.tracker_class(params)  # 根据具体的跟踪器类实例化对应的跟踪器模型
        
        return tracker

    def run_sequence(self, seq, visualization=None, debug=None, visdom_info=None, multiobj_mode=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()  # 获取具体跟踪器模型所需的参数（TrackerParams类）
        tracker = self.create_tracker(params)  # 根据读取的跟踪器参数创建指定跟踪器模型（不是跟踪器类，而是具体的跟踪器模型）
        
        visualization_ = visualization  # 是否可视化跟踪过程
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)  # 使用params中的debug参数
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, 'visualization', False)  # 使用params中的debug参数
            else:
                visualization_ = True if debug else False  # 有debug就会有可视化
        params.visualization = visualization_
        params.debug = debug_
        if visualization_:
            self._init_visdom(visdom_info, debug_)      # 初始化visdom可视化工具
            if self.visdom is None:  # 若初始化visdom失败（self.visdom=None）则使用matplotlib可视化
                self.init_visualization()
            tracker.visdom = self.visdom
        
        # Get init information
        init_info = seq.init_info() # 获得初始帧信息得到Dict类，包含了bbox等信息（seq是一个Sequence类）
        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        output = {'target_bbox': [],
                  'time': [],
                  'segmentation': [],
                  'object_presence_score': [],
                  'search_bbox':[],
                  'max_response': []}  # 要保存的结果的格式

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():  # 根据定义的output格式，提取出跟踪结果tracker_out中key对应的数值，若无该key，则值为None
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:  # 如果跟踪结果tracker_out中存在该key，或者值不为None，则将其存入格式化的结果output中
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])  # 读取首帧图片（RGB格式）

        if tracker.params.visualization and self.visdom is None:
            self.visualize(image, init_info.get('init_bbox'))  # 如果没有用到visdom工具，则使用matplotlib可视化图片
        start_time = time.time()
        out = tracker.initialize(image, init_info)  # 调用已经实例化的具体跟踪器类(如ATOM类)的initialize函数，返回初始化ATOM类的用时
        if out is None:
            out = {}
        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}  # 记录初始帧的结果
        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=2):
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                # else:
                #     time.sleep(0.1)

            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num-1)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)  # 调用已经实例化的具体跟踪器类(如ATOM类)的track函数，返回跟踪结果(Dict类)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            
            vis_bboxes = [out['target_bbox']]
            
            if self.show_search_region and 'search_bbox' in out:
                vis_bboxes.append(out['search_bbox'])
            if self.show_response_region and 'receptive_bbox' in out:
                vis_bboxes.append(out['receptive_bbox'])
                
            if self.visdom is not None:
                # tracker.visdom_draw_tracking(image, out['target_bbox'])
                tracker.visdom_draw_tracking(image, vis_bboxes)
            elif tracker.params.visualization:
                self.visualize(image, out['target_bbox'])

        for key in ['target_bbox']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        output['image_shape'] = image.shape[:2]

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the video file.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': OrderedDict({1: box}), 'init_object_ids': [1, ], 'object_ids': [1, ],
                    'sequence_object_ids': [1, ]}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox'][1]]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_webcam(self, debug=None, visdom_info=None):
        """Run the tracker with the webcam.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.new_init = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'init'
                    self.new_init = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        next_object_id = 1
        sequence_object_ids = []
        prev_output = OrderedDict()
        while True:
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                # else:
                #     time.sleep(0.1)
            
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            info = OrderedDict()
            info['previous_output'] = prev_output

            if ui_control.new_init:
                ui_control.new_init = False
                init_state = ui_control.get_bb()

                info['init_object_ids'] = [next_object_id, ]
                info['init_bbox'] = OrderedDict({next_object_id: init_state})
                sequence_object_ids.append(next_object_id)

                next_object_id += 1

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)

            if len(sequence_object_ids) > 0:
                info['sequence_object_ids'] = sequence_object_ids
                out = tracker.track(frame, info)
                prev_output = OrderedDict(out)

                if 'target_bbox' in out:
                    for obj_id, state in out['target_bbox'].items():
                        state = [int(s) for s in state]
                        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                     _tracker_disp_colors[obj_id], 5)
                        
                if self.show_search_region and 'search_bbox' in out:
                    for obj_id, state in out['search_bbox'].items():
                        state = [int(s) for s in state]
                        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                    (0, 0, 255), 5)
                        
                if self.show_response_region and 'receptive_bbox' in out:
                    for obj_id, state in out['receptive_bbox'].items():
                        state = [int(s) for s in state]
                        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                    (127, 255, 255), 5)

                vis_bboxes = list(out['target_bbox'].values())
                if self.show_search_region and 'search_bbox' in out:
                    vis_bboxes.extend(out['search_bbox'].values())
                if self.show_response_region and 'receptive_bbox' in out:
                    vis_bboxes.extend(out['receptive_bbox'].values())
                if self.visdom is not None:
                    tracker.visdom_draw_tracking(frame, vis_bboxes, None)
                
            # Put text
            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tongji V4R', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, 'Select target', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, 'r: reset', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'q: quit', (20, 105), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 's: search region', (20, 130), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'x: response region', (20, 155), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                next_object_id = 1
                sequence_object_ids = []
                prev_output = OrderedDict()

                info = OrderedDict()

                info['object_ids'] = []
                info['init_object_ids'] = []
                info['init_bbox'] = OrderedDict()
                tracker.initialize(frame, info)
                ui_control.mode = 'init'
            elif key == ord('s'):
                self.show_search_region = not self.show_search_region
            elif key == ord('x'):
                self.show_response_region = not self.show_response_region

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('parameter.{}.{}'.format(self.name, self.parameter_name))  # 根据跟踪器名称和对应参数文件加载对应的参数模块
        params = param_module.parameters()
        if self.hyper_params is not None:
            params.set_new_values(self.hyper_params)  # 设置超参值
            
        return params  # 返回TrackerParams类


    def init_visualization(self):
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state, segmentation=None):
        self.ax.cla()
        self.im_fig = self.ax.imshow(image)
        if segmentation is not None:
            self.ax.imshow(segmentation, alpha=0.5)

        if isinstance(state, (OrderedDict, dict)):
            boxes = [v for k, v in state.items()]
        else:
            boxes = (state,)

        for i, box in enumerate(boxes, start=1):
            col = _tracker_disp_colors[i]
            col = [float(c) / 255.0 for c in col]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=col, facecolor='none')
            self.ax.add_patch(rect)

        if getattr(self, 'gt_state', None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        draw_figure(self.fig)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def _read_image(self, image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)



