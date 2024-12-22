import time
import secrets

def generate_session_id():
    """
    生成一个基于日期和时间戳的唯一 32 位整数会话 ID，确保在同一天内不重复
    :return: 唯一的会话 ID 整数
    """
    now = time.localtime()
    date_str = time.strftime("%Y%m%d", now)

    date_int = int(date_str)
    random_number = secrets.randbelow(4096)

    session_id = (date_int << 12) | random_number
    return session_id