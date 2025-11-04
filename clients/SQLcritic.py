'''
SQL crtiticize client.

This client is designed to make sure that the SQL query is safe and reasonable to be executed.

It uses the already generated Vector-Database, schemaDB, which stores the schema (field) name, and the data strucutre (example)
of the field. By using the Vector-Database, it checks whether the SQL query uses correct filtering condition, including the key
words following:
    - WHERE
    - HAVING
    - ON
    - GROUP_BY

For example, if the field is in a JSON format or TEXT type, it would be less reasonable to use math operators (>=, <=, !=) for
filtering. The SQL critic should give a suggestion back to the CRICanalysisClient, helping the original client to generate a
proper SQL query instead.

Most importantly, it checks whether the SQL would modify the database or not. This can increase the safety and reliability of
the AI-agent.

The two-agent discussion workflow is like this:

Question ====> CRICanalysisClient -------> SQL query  ======> execute SQL =======> LLM =======> Answer
                      Λ                        |                Λ         context
                      |                        |                |
                      |         false          V         true   |
                 suggestions  <------- SQLcriticClient --------->

Rui Xue, r.xue@cern.ch
Nov, 2025
'''

import sys
import os
import re
from pathlib import Path
from sql_metadata import Parser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

curr_dir = Path(__file__).parent
target_dir = curr_dir.parent / "resources"
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory = str(target_dir / "schemaDB"), embedding_function = embedder)

# fetch all schema names as fields
fields = set()
for meta in db.get()["metadatas"]:
    field = meta.get("field")
    if field:
        fields.add(field)
fields = sorted(list(fields))


def FilterFields(query: str) -> list[str]:
    '''
    A simple and lightweight helper function that extracts
    fields used for filtering.
    '''
    parser = Parser(query)
    
    
    fields = set()
    for key in ("where", "having", "on", "group_by"):
        ff = parser.columns_dict.get(key, [])
        for token in ff:
            if re.match(r"^['\"].*['\"]$", token):   # no 'text' or "text" as fields
                continue
            if re.match(r"^[0-9.]+$", token):        # no pure number as fields
                continue
            if "%" in token:                         # no LIKE pattern as fields
                continue
            if token == "*":                         # no * as fields
                continue
            if '.' in token:                         # table.field => field
                fields.add(token.split('.')[-1])
            else:
                fields.add(token)
    fields = list(fields)
    return fields

class SQLcriticClient():

    def __init__(self, question:str, query:str):
        self.question = question
        self.query    = query
        self.schema_list = fields

    def extract_fields(self, query:str) -> list[str]:
        fields = FilterFields(query)
        contents = []
        for field in fields:
            # print(field)
            result = db.get(where={"field": field})
            if result["ids"]:
                contents.append(result["documents"][0])
        return contents

    def criticize(self) -> tuple[bool, str]:
        '''
        A loose criticize helper function that mainly verifies:
            1. Is the query safe? (Cannot modify the original database.)
            2. Is the query fetching correct? (It doesn't make sense using math relations to verify TEXT datatype.)
            3. It does not check whether the SQL can fully answer the question. (This is why I call it loose.)

        Finally, it outputs
            True, None
        or
            False, reasons...
        '''

        field_contents = self.extract_fields(self.query)
        prompt = f"""
        You are an expert on ATLAS PanDA CRIC (Computing Resource Information Catalogue) database system.
        Your job is to review the SQL query and verify whether it is appropriate and safe to answer the user's question.

        ---
        Question from the user:
        {self.question}

        SQL query generated:
        {self.query}

        Database schema (table name, field name, data type, and examples if available):
        {"\n".join(field_contents)}

        All available schema:
        {",".join(self.schema_list)}
        ---

        Your tasks:

        0. **Safety check**: 
        - Verify the SQL will NOT modify the database (no INSERT, UPDATE, DELETE, DROP, ALTER, etc.).

        1. **General question detection**:
        - If the question is **general, descriptive, or definition-like**, such as:
            - starts with “what is”, “explain”, “describe”, or “show”
            - or asks for an overview or description (not a specific dataset)
        - Then, even if the SQL query is not directly answering it, as long as it is safe and queries CRIC-related data,
            you should **approve it** by returning:
            → "True, None"

        2. **Structured data verification** (only apply if NOT a general question):
        - Check if the filtering fields in the SQL exist in the schema.
        - Verify if filtering logic and operators are reasonable based on the example data.

        3. **Flexibility**
        - CRIC schema fields such as 'region', 'countrygroup', or 'cloud' **do not represent continents or world regions**.
        They are internal ATLAS operational groupings, not geography. Therefore:
        - Do NOT expect them to filter countries by continent (like 'North America', 'Europe', 'Asia').
        - If the user question includes geographic concepts not encoded in the schema, simply approve the query.

        - If the query is syntactically correct, uses only existing fields, and is safe,
        approve it even if it does not fully match every part of the natural-language request.
          
        4. **Final output**:
        - If all checks pass → return "True, None"
        - If any check fails → return "False, ..." with reasons (no need to rewrite SQL)

        ---

        Output format requirement:
        Because I will take the output and deal with  ans = tuple(map(str, output.text.strip().split(",")[1]))
        Return your answer strictly as a string:
        True,None   or   False,reason ...
        
        No extra commentary.
        """
        response = gemini_model.generate_content(prompt)
        ans = tuple(map(str, response.text.strip().split(",",1)))
        _bool = ans[0].strip().lower() == "true"
        _reason = ans[1].strip()
        return (_bool, _reason)


def test():
    # query = '''
    # select distinct site from queuedata where copytools = "rucio"
    # '''
    # question = "which sites are using the rucio copytool?"
    

    # query = '''
    # delete distinct country from queuedata where tier_level=1
    # '''
    # question = '''
    # please help me to delete out all countries that hold tier_level=1 sites.
    # '''
    
    query = '''
    select distinct country from queuedata where tier_level=1
    '''
    question = '''
    please help me to select out all countries that hold tier_level=1 sites.
    '''
    # query = '''
    # select panda, country from queuedata
    # '''
    
    # question = '''
    # what is panda system? please use cric data to help explain
    # '''
    print("Test Question: ", question)
    print("Test Query: ", query)
    crtic_client = SQLcriticClient(question, query)
    print(crtic_client.criticize())

if __name__ == "__main__":
    test()