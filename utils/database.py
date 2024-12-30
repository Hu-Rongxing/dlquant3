from psycopg2 import pool
from contextlib import contextmanager
from sqlalchemy import create_engine, text
import urllib.parse
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Optional, Union, Tuple
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

# 自定义
# 配置日志
from logger import log_manager
applogger = log_manager.get_logger(__name__)


class DatabaseConfigError(Exception):
    """数据库配置异常"""
    pass


class PGConfig:
    """PostgreSQL数据库配置类"""

    def __init__(self, database: str = 'quant'):
        """
        初始化PostgreSQL数据库配置

        Args:
            database (str): 数据库名称，默认为'quant'
        """
        try:
            self.host = self._get_config("database.pg_host")
            self.port = self._get_config("database.pg_port")
            self.database = database
            self.username = self._get_config("database.pg_user")
            self.password = self._get_config("database.pg_password")
        except KeyError as e:
            raise DatabaseConfigError(f"缺少必要的数据库配置: {e}")

    def _get_config(self, key: str) -> str:
        """
        获取配置，可以根据实际情况修改

        Args:
            key (str): 配置键

        Returns:
            str: 配置值
        """
        from config import settings
        value = settings.get(key)
        if not value:
            raise KeyError(f"配置 {key} 未找到")
        return value

    def get_connection_params(self) -> Dict[str, Union[str, int]]:
        """
        获取连接参数字典

        Returns:
            Dict[str, Union[str, int]]: 连接参数
        """
        return {
            'database': self.database,
            'host': self.host,
            'user': self.username,
            'password': self.password,
            'port': self.port,
            'client_encoding': 'utf-8'
        }

class PGConnectionPool:
    """PostgreSQL连接池管理"""

    def __init__(self, database: str = 'quant',
                 min_connections: int = 1,
                 max_connections: int = 10,
                 timeout: int = 30):
        # 增加超时和重试机制
        self.timeout = timeout
        self._pool = None
        self._create_connection_pool(min_connections, max_connections)

    def _create_connection_pool(self, min_connections: int, max_connections: int):
        try:
            # 增加连接超时和重试逻辑
            self._pool = pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                **self.config.get_connection_params(),
                connect_timeout=self.timeout
            )
        except Exception as e:
            applogger.error(f"创建连接池失败: {e}")
            # 可以考虑实现自动重试机制
            raise

    @contextmanager
    def connection(self):
        """
        上下文管理器，获取和释放连接

        Yields:
            psycopg2 connection
        """
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            applogger.error(f"数据库连接错误: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """
        执行查询

        Args:
            query (str): SQL查询语句
            params (Optional[Tuple]): 查询参数

        Returns:
            List[Tuple]: 查询结果
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()


    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def execute_insert(self, query: str, data: Union[Dict, List[Dict]]) -> Optional[int]:
        """
        执行插入操作

        Args:
            query (str): 插入SQL语句
            data (Union[Dict, List[Dict]]): 插入数据

        Returns:
            Optional[int]: 最后插入行的ID
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                if isinstance(data, dict):
                    cur.execute(query, data)
                elif isinstance(data, list):
                    cur.executemany(query, data)
                conn.commit()
                return cur.lastrowid


def create_sqlalchemy_engine(
        database: str = 'quant',
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
):
    """
    创建SQLAlchemy引擎，增强错误处理和编码兼容性
    """
    try:
        # 从配置读取默认值
        config = PGConfig(database)

        # 使用传入参数或配置中的值
        db_host = host or config.host
        db_port = port or config.port
        db_username = username or config.username
        db_password = password or config.password

        # 处理可能的编码问题
        def sanitize_connection_param(param: str) -> str:
            """清理连接参数中的特殊字符"""
            try:
                # 尝试解码和重新编码
                return param.encode('utf-8', errors='ignore').decode('utf-8')
            except Exception as e:
                logging.warning(f"参数清理失败：{e}")
                return param

        # URL编码密码
        encoded_password = urllib.parse.quote_plus(
            sanitize_connection_param(db_password)
        )

        # 构建详细的连接字符串
        connection_string = (
            f"postgresql+psycopg2://{sanitize_connection_param(db_username)}:"
            f"{encoded_password}@{sanitize_connection_param(db_host)}:"
            f"{db_port}/{sanitize_connection_param(database)}"
            f"?client_encoding=utf8"
        )

        # 创建引擎，增加更多连接参数
        engine = create_engine(
            connection_string,
            pool_size=10,  # 连接池大小
            max_overflow=20,  # 最大溢出连接数
            pool_timeout=30,  # 连接超时
            pool_recycle=1800,  # 连接回收时间
            pool_pre_ping=True,  # 连接健康检查

            # 额外的连接参数
            connect_args={
                'keepalives': 1,  # 保持连接活跃
                'keepalives_idle': 30,  # 空闲时间
                'keepalives_interval': 10,  # 检查间隔
                'keepalives_count': 5,  # 重试次数
            }
        )

        # 测试连接 - 使用 text() 包装 SQL 语句
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        logging.info("数据库连接成功")
        return engine

    except SQLAlchemyError as e:
        logging.error(f"SQLAlchemy引擎创建失败：{e}")
        raise
    except Exception as e:
        logging.error(f"数据库连接未知错误：{e}")
        raise
    # 使用示例


def main():
    # 连接池使用示例
    pg_pool = PGConnectionPool()

    # 查询
    results = pg_pool.execute_query("SELECT * FROM users WHERE id = %s", (1,))

    # 插入
    insert_query = "INSERT INTO users (name, email) VALUES (%(name)s, %(email)s)"
    user_data = {'name': 'John', 'email': 'john@example.com'}
    pg_pool.execute_insert(insert_query, user_data)

    # SQLAlchemy 引擎使用
    engine = create_sqlalchemy_engine()
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM users"))
        for row in result:
            print(row)


if __name__ == '__main__':
    main()