import sqlite3
import pandas as pd
from .sorted_ambrosia import all_queries

ambrosia_queries = [
    "What safety measures will the architect require?",
    "Show me the students and what they do?",
    "What functions are there performed by all the HR professionals?",
    "What is the salary of a role that each hospital has?",
    "What can help students prepare for testing?",
    "Tell me the systems every support specialist works with.",
    "What is the background of the photojournalist in the table?",
    "Show me the mortality trends and population for new york",
    "What is the state of the system that each IT support specialist works with?",
    "Who heads the bank with the largest withdrawal?",
    "What experience do I need for the open position with the greatest salary?",
    "Describe the benefit we have for every employer.",
    "What is the percentage distribution of annual revenue per occupational category?",
    "Show me the roles every hospital has.",
    "Which context has the most plays set in it, and how many?",
    "Which resource is Alice Smith using?",
    "What is the name and phone number of least experienced applicant.",
    "Provide latest news and news analysis on the Turkey-Syria Earthquakes.",
    "Show me the name and id of the lead actor with the most films",
    "What are the demographics of Author B's readers?",
    "Which posted jobs require which qualifications?"
]

def build_ambrosia_prompt(row) -> str:
    return f"""Given the following database dump:
{row['db_dump']}
and the following question:
{row['question']}
generate the SQL query and output ONLY the SQL query and nothing else."""

def find_row_by_question(df: pd.DataFrame, question: str) -> pd.Series:
    for _, row in df.iterrows():
        if row['question'].strip().lower() == question.strip().lower():
            return row
    raise ValueError(f"Question not found in DataFrame: {question}")


def execute_sql_query(cursor, query: str):
    """Execute a SQL query and return the results.

    Args:
        cursor: SQLite database cursor.
        query (str): SQL query to execute.
    Returns:
        list: Results of the SQL query.
    """
    try:
        cursor.execute(query)
        return True, cursor.fetchall()
    except Exception as e:
        return False, [str(e)]

def test_ambrosia_sql_queries(sql_dump: str, queries: list[str], golden_queries: str):
    """Test Ambrosia SQL queries against an in-memory SQLite database.

    Args:
        sql_dump (str): The SQL dump to initialize the database.
        queries (list[str]): List of SQL queries to test.
    """

    # preprocess the queries
    queries = [q.strip() for q in queries]
    # some models return a codeblock, ie, ```sql\n...\n``` so we remove it
    queries = [q.replace("```sql\n", "").replace("\n```", "") for q in queries]

    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.executescript(sql_dump)

    try:
        golden_queries = eval(golden_queries)
        golden_results = set([row for q in golden_queries for row in execute_sql_query(cursor, q)[1]])
    except Exception as e:
        golden_queries = set()
        golden_results = set()
        print(f"Error evaluating golden queries: {e}")
        import traceback
        traceback.print_exc()

    results = []
    for i, q in enumerate(queries):
        success, res_or_error = execute_sql_query(cursor, q)
        if success:
            results.append({
                'index': i,
                'query': q,
                'is_runnable': True,
                'result': res_or_error,
                'is_golden': any([r in golden_results for r in res_or_error])
            })
        else:
            results.append({
                'index': i,
                'query': q,
                'is_runnable': False,
                'error': res_or_error[0],
            })

    return results


def load_ambrosia_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def init_stats_file(stats_output_file: str):
    from pathlib import Path
    stats_path = Path(stats_output_file)
    if not stats_path.exists():
        with open(stats_path, "w") as f:
            f.write("model,question_id,question,temperature,unique_queries,runnable,unique_results,golden_percent\n")


def get_ambrosia_prompts(mode: str, df: pd.DataFrame) -> list[tuple[str, int | None]]:
    if mode == "ambrosia":
        return [(build_ambrosia_prompt(row), row.name) for _, row in df.iterrows()]
    elif mode == "ambrosia-short":
        prompt_list = [(build_ambrosia_prompt(row), row.name) for _, row in df.iterrows()]
        return prompt_list[-20:]
    elif mode == "ambrosia-selected":
        return [
            (build_ambrosia_prompt(
                (row := find_row_by_question(df, q))), None) for q in ambrosia_queries[:1]
        ]
    elif mode == "ambrosia-top5":
        return [
            (build_ambrosia_prompt(
                (row := find_row_by_question(df, q))), row.name) for q in ambrosia_queries[:5]
        ]
    elif mode == "ambrosia-top25":
        return [
            (build_ambrosia_prompt(
                (row := find_row_by_question(df, q))), row.name) for q in ambrosia_queries[:25]
        ]
    elif mode == "ambrosia-top100":
        return [
            (build_ambrosia_prompt(
                (row := find_row_by_question(df, q))), row.name) for q in all_queries[:100]
        ]
    elif mode == "ambrosia-top500":
        return [
            (build_ambrosia_prompt(
                (row := find_row_by_question(df, q))), row.name) for q in all_queries[:500]
        ]
    return []


def compute_ambrosia_coverage(sql_dump, generated_queries, golden_queries_str):
    """Compute the percentage of golden query intents covered by the generated queries."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.executescript(sql_dump)

    # Parse golden queries
    g_sqls = eval(golden_queries_str)
        
    if not g_sqls:
        raise ValueError("No golden queries found.")

    # Get golden result signatures (intents)
    golden_sigs = set()
    for q in g_sqls:
        success, res = execute_sql_query(cursor, q)
        if success:
            sig = tuple(res)
            golden_sigs.add(sig)
    
    assert len(golden_sigs) != 0

    # Normalize generated queries
    gen_queries = [q.strip().replace("```sql\n", "").replace("\n```", "") for q in generated_queries]

    # Check coverage
    covered_sigs = set()
    for q in gen_queries:
        success, res = execute_sql_query(cursor, q)
        if success:
            sig = tuple(res)
            if sig in golden_sigs:
                covered_sigs.add(sig)
                
    return len(covered_sigs) / len(golden_sigs)


def compute_ambrosia_stats(results, temperatures, prompt_list, df, model_name, stats_output_file=None, metric="coverage"):
    print(f"Computing ambrosia stats on full results (metric={metric})...")
    for T in temperatures:
        temp_bucket = results[T]
        for prompt_text, prompt_id_val in prompt_list:
             entry = temp_bucket.get(prompt_text)
             if entry and isinstance(entry, (list, tuple)) and len(entry) >= 1:
                 # entry is (raw_sents, raw_lps, raw_t1_lps, raw_lengths)
                 raw_sents = entry[0]

                 prompt_id = int(prompt_id_val) if prompt_id_val is not None else None
                 df_row = df.iloc[prompt_id] if prompt_id is not None else None

                 if df_row is not None:
                     sql_dump = df_row['db_dump']
                     golden_queries = df_row['ambig_queries']
                     deduped_generations = set(raw_sents)
                     if not deduped_generations:
                         continue

                     if metric == "coverage":
                         golden_score = compute_ambrosia_coverage(
                             sql_dump,
                             list(deduped_generations),
                             golden_queries
                         )
                         results_stats = test_ambrosia_sql_queries(
                             sql_dump,
                             list(deduped_generations),
                             golden_queries
                         )
                         runnable_count = sum(1 for r in results_stats if r['is_runnable'])
                         list_of_sql_results = [r['result'] for r in results_stats if r['is_runnable']]
                         len_unique_results = len(set(tuple(res) for res in list_of_sql_results))
                         
                         score_val = golden_score * 100.0 # Convert to percent scale like the old metric
                         
                     else: # metric == "accuracy" (OLD BEHAVIOR)
                         results_stats = test_ambrosia_sql_queries(
                             sql_dump,
                             list(deduped_generations),
                             golden_queries
                         )
                         runnable_count = sum(1 for r in results_stats if r['is_runnable'])
                         list_of_sql_results = [r['result'] for r in results_stats if r['is_runnable']]
                         len_unique_results = len(set(tuple(res) for res in list_of_sql_results))
                         
                         score_val = sum(1 for r in results_stats if r.get('is_golden', False)) / len(deduped_generations) * 100.0 if deduped_generations else 0.0

                     print(f"  Question {prompt_id} T={T}: generated {len(deduped_generations)} unique SQL queries, {runnable_count} runnable, {len_unique_results} unique results, {score_val:.2f}% ({metric}).")

                     if stats_output_file:
                         with open(stats_output_file, "a") as f:
                             escaped_question = df_row['question'].replace('"', '""')
                             f.write(f'{model_name},{prompt_id},"{escaped_question}",{T},{len(deduped_generations)},{runnable_count},{len_unique_results},{score_val:.4f}\n')


if __name__ == "__main__":
    sql_dump = """
    CREATE TABLE restaurants (id INTEGER PRIMARY KEY, name TEXT, cuisine TEXT, capacity INTEGER);
    INSERT INTO restaurants (name, cuisine, capacity) VALUES
        ('Luigi''s', 'Italian', 150),
        ('Pasta Palace', 'Italian', 200),
        ('Sushi Central', 'Japanese', 100),
        ('Curry House', 'Indian', 250);
    """

    queries = [
        "SELECT name FROM restaurants WHERE cuisine='Italian' AND capacity>=180;",
        "SELECT name FROM restaurants WHERE cuisine='Mexican';",  # No Mexican restaurants
        "SELECT COUNT(*) FROM restaurants;",  # Valid query
        "SELEC name FROM restaurants;",  # Invalid SQL syntax
    ]

    r = test_ambrosia_sql_queries(sql_dump, queries)
    print(*r, sep="\n")
