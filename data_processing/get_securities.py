import pandas as pd
from utils.database import create_sqlalchemy_engine, PGConnectionPool
from sqlalchemy import text


def get_investment_target() -> pd.DataFrame:
    query = """  
        SELECT * FROM investment_target   
        WHERE status=true  
        """
    pg_pool = PGConnectionPool()

    with pg_pool.connection() as conn:
        df = pd.read_sql_query(
            query,
            conn
        )
    return df


if __name__ == '__main__':
    investment_target_df = get_investment_target()
    print(investment_target_df)


