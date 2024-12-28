import pandas as pd
from utils.database import create_sqlalchemy_engine, PGConnectionPool
from sqlalchemy import text


def get_investment_target() -> pd.DataFrame:
    query = """  
        SELECT * FROM investment_target   
        WHERE status=true  
        """
    engine = create_sqlalchemy_engine()
    df = pd.read_sql_query(query, engine)

    return df


if __name__ == '__main__':
    investment_target_df = get_investment_target()
    print(investment_target_df)


