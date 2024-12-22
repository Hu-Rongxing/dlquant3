from email.mime.text import MIMEText  # 导入 MIMEText 类，用于构建电子邮件内容
import smtplib
# 自定义
from config import settings

def send_email_notification(message):
    """
    发送电子邮件通知。

    :param message: 要发送的消息内容，可以是字符串或 MIMEText 对象
    """
    try:
        # 邮件设置
        sender_email = settings.get("qqmail.SMTP_USER_NAME", "hurongxing@vip.qq.com")  # 发件人电子邮件地址
        receiver_email = settings.get("qqmail.ALERT_EMAIL_ADDRESS", "280712999@qq.com")  # 收件人电子邮件地址
        smtp_host = settings.get("qqmail.SMTP_HOST", "smtp.qq.com")
        smtp_port = int(settings.get("qqmail.SMTP_PORT", 587))
        password = settings.get("qqmail.SMTP_PASSWORD", "")
        subject = "交易系统日志"  # 邮件主题

        # 判断 message 的类型
        if isinstance(message, str):
            # 如果是字符串，构建 MIMEText 对象，指定为 HTML 格式
            html_message = f"""  
            <html>  
                <body>  
                    <h1>{subject}</h1>  
                    <p>{message}</p>  
                    <p>----------</p>  
                </body>  
            </html>  
            """
            msg = MIMEText(html_message, 'html')
        elif isinstance(message, MIMEText):
            # 如果是 MIMEText 对象，直接使用
            msg = message
        else:
            raise ValueError("message 必须是字符串或 MIMEText 对象")

        msg['Subject'] = subject  # 设置邮件主题
        msg['From'] = sender_email  # 设置发件人
        msg['To'] = receiver_email  # 设置收件人

        # print(f"邮件内容: {msg.as_string()}")  # 打印邮件内容

        # 显式创建 SMTP 连接
        smtp = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender_email, password)
        smtp.sendmail(sender_email, receiver_email, msg.as_string())
        smtp.quit()  # 关闭 SMTP 连接

        print("成功发送邮件。")  # 打印成功消息
    except Exception as e:
        print(f"发送邮件失败: {e}")  # 捕获并打印发送邮件时的异常


if __name__ == '__main__':
    # 测试发送 HTML 格式的邮件
    html_message = """文本内容"""
    send_email_notification(html_message)

    # 测试发送 MIMEText 对象
    test_msg = MIMEText("<h1>这是一个测试邮件，使用 MIMEText 对象。</h1>", 'html')
    send_email_notification(test_msg)