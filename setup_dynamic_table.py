from operator import itemgetter

from dotenv import load_dotenv
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-70b-versatile")


def get_table_details():
    """
    :return: A concatenated string containing the details of the tables. The string includes table names and their descriptions, formatted with each table's information on a new line.
    """
    # Read the CSV file into a DataFrame
    table_description = pd.read_csv(filepath_or_buffer="./file/desrciption_db.csv")
    table_docs = []

    # Iterate over the DataFrame rows to create Document objects
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table Name'] + "\n" + "Table Description:" + row[
            'Description'] + "\n\n"

    return table_details


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_details = get_table_details()
table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
                        The tables are:

                        {table_details}

                        Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.
                        """


def get_tables(tables: List[Table]) -> List[str]:
    """
    :param tables: List of Table objects that need to be processed
    :return: List of table names extracted from the provided Table objects
    """
    tables = [table.name for table in tables]
    return tables


select_table = ({"input": itemgetter("question")}
                | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)
                | get_tables
                )

