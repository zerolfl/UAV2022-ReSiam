import smtplib
import os
import pandas as pd
import re

from email.header import Header
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from .get_system_info import PC_info

class ExperimentMail:
    def __init__(self, subject=None, note=None, file_path=[], all_hyper_params_dict=None):
        self.subject = subject  # 跟踪器+实验名
        self.note = note
        self.file_path = file_path
        self.all_hyper_params_dict = all_hyper_params_dict
        
        self.pc = PC_info()
        # 第三方 SMTP 服务
        self.mail_host = "smtp.qq.com"  # 填写邮箱服务器:这个是qq邮箱服务器，直接使用smtp.qq.com
        self.sender = ''  # 填写邮箱地址
        self.mail_pass = ""  # 填写在qq邮箱设置中获取的授权码
        self.receivers = ['']  # 填写收件人的邮箱，QQ邮箱或者其他邮箱，可多个，中间用,隔开

    def attach_file(self, file_path):
        """添加附件(CSV, PDF, ...)"""
        file = MIMEApplication(open(file_path, 'rb').read())
        file.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file_path))
        self.message.attach(file)
        
    def attach_text(self, content, type='html', charset='utf-8'):
        """添加文本"""
        content = MIMEText(content, type, charset)
        self.message.attach(content)
    
    def gen_basic_info(self):
        # 生成基本的邮件信息
        message = MIMEMultipart('mixed')  # 设置附件模式
        message['Subject'] = Header('{} ({})'.format(self.subject, self.note), 'utf-8')  # 邮件标题
        host_name = self.pc.hostname
        message['From'] = Header(host_name, 'utf-8')  # 发件人名称 - 主机名
        # message['To'] =  Header("测试", 'utf-8')  # 收件人名称
        
        # 添加正文
        content = '''
        <p>
            Tracker-Experiment: {}<br/>
            Note: {}<br/>
            Device: {}<br/>
                <span style='margin-left:2em'>System: {}</span><br/>
                <span style='margin-left:2em'>CPU: {}</span><br/>
                <span style='margin-left:2em'>RAM: {}</span><br/>
                <span style='margin-left:2em'>GPU: {}</span>
        </p>
        '''.format(self.subject, self.note, self.pc.hostname, self.pc.platform, self.pc.cpu, self.pc.ram, self.pc.gpu)
        
        content_msg = MIMEText(content, 'html', 'utf-8')
        message.attach(content_msg)
        
        self.message = message
        
    def send(self):
        self.gen_basic_info()
        
        if self.all_hyper_params_dict is not None:
            _hp_info = pd.DataFrame.from_dict(self.all_hyper_params_dict, orient='index')
            _hp_info.columns = range(1,len(_hp_info.columns)+1)
            title = 'Experiment Hyper Params'
            hp_info = _optim_html_table(_hp_info.to_html(), title)
            hp_info = hp_info.replace('NaN', '').replace('None', '')
            self.attach_text(hp_info)
        
        self.file_path = self.file_path if isinstance(self.file_path, list) else [self.file_path]
        for file_path in self.file_path:
            self.attach_file(file_path)
            if '.csv' in file_path:
                _df = pd.read_csv(file_path)
                _df.index=_df.index+1
                
                _df = _df[_df.columns.drop(list(_df.filter(regex='FPS|Norm.DP')))]  # 指定不要的指标(删列), 采用正则表达
                _df = _df[list(_df.filter(regex='ID|AUC|DP'))]  # 指定不要的指标(删列), 采用正则表达
                
                title = '⭐Top{} Trackers'.format(_df[:10].shape[0])
                top10_res_info = _optim_html_table(_df[:10].to_html(), title)
                self.attach_text(top10_res_info)
                
        try:
            smtpObj = smtplib.SMTP_SSL(self.mail_host, 465)  # 建立smtp连接，qq邮箱必须用ssl边接，因此边接465端口
            smtpObj.login(self.sender, self.mail_pass)  # 登陆
            smtpObj.sendmail(self.sender, self.receivers, msg=self.message.as_string())  # 发送
            smtpObj.quit()
            # print('发送成功!')
        except smtplib.SMTPException as e:
            print('邮件发送失败!')


def _optim_html_table(df_html, title):
    _num_row = len([substr.start() for substr in re.finditer('<tr>', df_html)])  # 找到每一行
    df_html = "<div style=\"text-align: center; overflow: auto;\">" + df_html  # 保证内容居中
    df_html = df_html.replace(r'<table border="1" class="dataframe">', '<table style=\"border: 0.5px solid black; margin: 8px 8px; table-layout:fixed;\"> <caption><b>{}</b></caption>'.format(title))
    df_html = df_html.replace(r'<tr style="text-align: right;">', '<tr style=\"background-color: #015DAA; color: #ffffff;\">')  # 更换表头样式
    for i in range(_num_row):  # 找到每一行
        if i % 2 == 0:
            df_html = df_html.replace(r'<tr>', '<tr style=\"background-color: #ebf5ff; \">', 1)  # 奇数行
        else:
            df_html = df_html.replace(r'<tr>', '<tr style=\"background-color: #ffffff; \">', 1)  # 偶数行
    df_html += "</div>"
    return df_html


if __name__ == '__main__':
    
    hyper_params = {
                    'use_detection_sample': [False, True], 
                    'instance_side': [255, 287],
                    'train_skipping': [1, 5, 10],
                    }
    
    mail = ExperimentMail('RPN-HC', '40/120', all_hyper_params_dict=hyper_params)
    mail.send()
