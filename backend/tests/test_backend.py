from __future__ import annotations

import pandas as pd
import pytest

from app.loader import load_dataframe
from app.ollama_agent import _fallback_analysis, _fallback_summary
from app.profiling import profile_dataframe
from app.schemas import AnalysisPlan
from app.safe_exec import SafetyError, execute_analysis, validate_code


COLLEGE_COLUMNS = ["college", "state", "tuition", "type"]


def college_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "college": ["A College", "B College", "C College", "D College"],
            "state": ["CA", "CA", "NY", "TX"],
            "tuition": [100, 200, 300, 400],
            "type": ["Private", "Public", "Private", "Public"],
        }
    )


def test_profile_dataframe_reports_quality() -> None:
    df = pd.DataFrame({"state": ["CA", "CA", "NY"], "revenue": [10, None, 5], "constant": [1, 1, 1]})
    profile = profile_dataframe("test", "sales.csv", df, pd.Timestamp.utcnow().to_pydatetime())
    assert profile.metadata.row_count == 3
    assert profile.metadata.column_count == 3
    assert "constant" in profile.constant_columns
    revenue = next(column for column in profile.columns if column.name == "revenue")
    assert revenue.missing_count == 1


def test_load_json_array_dataset(tmp_path) -> None:
    path = tmp_path / "colleges.json"
    path.write_text('[{"college":"A","tuition":100},{"college":"B","tuition":200}]')
    df, name = load_dataframe(path)
    assert name == "colleges.json"
    assert list(df.columns) == ["college", "tuition"]
    assert len(df) == 2


def test_load_jsonl_dataset(tmp_path) -> None:
    path = tmp_path / "colleges.jsonl"
    path.write_text('{"college":"A","tuition":100}\n{"college":"B","tuition":200}\n')
    df, name = load_dataframe(path)
    assert name == "colleges.jsonl"
    assert list(df.columns) == ["college", "tuition"]
    assert len(df) == 2


def test_load_tsv_dataset(tmp_path) -> None:
    path = tmp_path / "colleges.tsv"
    path.write_text("college\ttuition\nA\t100\nB\t200\n")
    df, name = load_dataframe(path)
    assert name == "colleges.tsv"
    assert list(df.columns) == ["college", "tuition"]
    assert len(df) == 2


def test_safe_exec_rejects_imports() -> None:
    with pytest.raises(SafetyError):
        validate_code("import os\nresult = 1", ["a"])


def test_safe_exec_rejects_missing_columns() -> None:
    with pytest.raises(SafetyError):
        validate_code("result = df['missing'].sum()", ["present"])


def test_safe_exec_runs_groupby() -> None:
    df = pd.DataFrame({"state": ["CA", "CA", "NY"], "revenue": [10, 7, 5]})
    result, tables, charts = execute_analysis("result = df.groupby('state')['revenue'].sum().sort_values(ascending=False)", df)
    assert result["CA"] == 17
    assert tables[0].rows[0]["revenue"] == 17
    assert charts == []


def test_fallback_handles_dataset_overview_question() -> None:
    analysis = _fallback_analysis("What is this dataset about?", COLLEGE_COLUMNS)
    assert analysis.classification == "Dataset Information"
    assert analysis.plan.grounding_status == "Ready"
    assert analysis.code is not None
    result, tables, _ = execute_analysis(analysis.code, college_dataframe())
    answer = _fallback_summary(analysis.plan, result, True)
    assert result["row_count"] == 4
    assert result["dataset_profile"]["dataset_subject"] == "colleges or universities"
    assert result["dataset_profile"]["primary_entity"] == "College/University"
    assert result["dataset_profile"]["entity_count"] == 4
    assert "semantic_columns" in result["dataset_profile"]
    assert "statistics" in result["dataset_profile"]
    assert result["column_profiles"][1]["top_values"][0]["value"] == "CA"
    assert "colleges or universities" in answer
    assert "4 records" in answer
    assert "Primary entity: College/University" in answer
    assert tables[0].rows[0]["row_count"] == 4


@pytest.mark.parametrize(
    "question",
    [
        "What information does this dataset contain?",
        "Tell me about this dataset.",
        "Tell me about the data.",
        "Summarize this file.",
        "Explain what this table contains.",
        "What does this CSV include?",
        "What information is inside?",
        "Describe this file.",
        "Give me an overview.",
        "What kind of data is this?",
        "What info is in this file?",
    ],
)
def test_fallback_handles_generic_overview_variants(question: str) -> None:
    analysis = _fallback_analysis(question, COLLEGE_COLUMNS)
    assert analysis.classification == "Dataset Information"
    assert analysis.plan.grounding_status == "Ready"
    assert analysis.code is not None


def test_fallback_handles_domain_record_count_question() -> None:
    analysis = _fallback_analysis("How many colleges are there?", COLLEGE_COLUMNS)
    assert analysis.classification == "Dataset Information"
    assert analysis.plan.planned_operations == ["Count rows in the uploaded dataset", "Count distinct non-missing values in college"]
    assert analysis.code is not None
    result, _, _ = execute_analysis(analysis.code, college_dataframe())
    answer = _fallback_summary(analysis.plan, result, False)
    assert result["record_count"] == 4
    assert result["distinct_entity_count"] == 4
    assert "4 records" in answer


def test_fallback_handles_subset_count_question_from_values() -> None:
    analysis = _fallback_analysis("How many colleges are private?", COLLEGE_COLUMNS)
    assert analysis.classification == "Statistical Analysis"
    assert analysis.code is not None
    result, _, _ = execute_analysis(analysis.code, college_dataframe())
    assert result["matched_count"] == 2
    assert result["percent"] == 50.0
    assert "type" in result["matched_columns"]


def test_fallback_lists_matching_rows_from_value_question() -> None:
    analysis = _fallback_analysis("Which colleges are private?", COLLEGE_COLUMNS)
    assert analysis.classification == "Dataset Information"
    assert analysis.code is not None
    result, tables, _ = execute_analysis(analysis.code, college_dataframe())
    assert len(result) == 2
    assert {row["college"] for row in result} == {"A College", "C College"}
    assert "college" in tables[0].columns


def test_fallback_handles_unique_record_count_question() -> None:
    analysis = _fallback_analysis("How many unique colleges are there?", COLLEGE_COLUMNS)
    assert analysis.classification == "Dataset Information"
    assert analysis.code is not None
    result, _, _ = execute_analysis(analysis.code, college_dataframe())
    assert result["distinct_entity_count"] == 4


def test_fallback_handles_percentage_question() -> None:
    analysis = _fallback_analysis("What percentage of colleges are in each state?", COLLEGE_COLUMNS)
    assert analysis.classification == "Statistical Analysis"
    assert analysis.code is not None
    result, tables, _ = execute_analysis(analysis.code, college_dataframe())
    assert result[0]["value"] == "CA"
    assert result[0]["percent"] == 50.0
    assert tables[0].columns == ["value", "count", "percent"]


def test_fallback_handles_average_question() -> None:
    analysis = _fallback_analysis("What is the average tuition?", COLLEGE_COLUMNS)
    assert analysis.classification == "Statistical Analysis"
    assert analysis.code is not None
    result, _, _ = execute_analysis(analysis.code, college_dataframe())
    assert result["column"] == "tuition"
    assert result["value"] == 250.0
    answer = _fallback_summary(analysis.plan, result, False)
    assert "tuition" in answer
    assert "250.0" in answer


def test_fallback_handles_share_values_without_column_name() -> None:
    analysis = _fallback_analysis("What share are private vs public?", COLLEGE_COLUMNS)
    assert analysis.classification == "Statistical Analysis"
    assert analysis.code is not None
    result, tables, _ = execute_analysis(analysis.code, college_dataframe())
    type_rows = [row for row in result if row["column"] == "type"]
    assert {"Private", "Public"} == {row["value"] for row in type_rows}
    assert all(row["percent"] == 50.0 for row in type_rows)
    assert tables[0].columns == ["column", "value", "count", "percent"]


def test_fallback_summarizes_single_value_dict_result() -> None:
    plan = AnalysisPlan(
        category="Statistical Analysis",
        question_understanding="Which college has the most enrollment?",
        selected_columns=["college", "enrollment"],
        column_selection_reason="Matched college and enrollment columns.",
        planned_operations=["Group by college", "Sum enrollment", "Return top college"],
        output_type="Text + Table",
        estimated_complexity="Low",
        grounding_status="Ready",
        status="Ready to Execute",
    )
    result = {"A College": 1234}
    answer = _fallback_summary(plan, result, False)
    assert "A College" in answer
    assert "1234" in answer


def test_fallback_lists_some_colleges_from_named_column() -> None:
    analysis = _fallback_analysis("list some colleges from the dataset", COLLEGE_COLUMNS)
    assert analysis.classification == "Dataset Information"
    assert analysis.code is not None
    result, tables, _ = execute_analysis(analysis.code, college_dataframe())
    assert result[0]["college"] == "A College"
    assert tables[0].columns == ["college"]


@pytest.mark.parametrize("question", ["show a few records", "provide some examples from the data", "name a few entries"])
def test_fallback_lists_examples_from_broad_list_intents(question: str) -> None:
    analysis = _fallback_analysis(question, COLLEGE_COLUMNS)
    assert analysis.classification == "Dataset Information"
    assert analysis.code is not None
    result, _, _ = execute_analysis(analysis.code, college_dataframe())
    assert result


def test_overview_expands_college_abbreviations_and_ignores_unnamed_label() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": ["Abilene Christian University", "Baylor University"],
            "Private": ["Yes", "Yes"],
            "Apps": [1660, 6075],
            "Accept": [1232, 5349],
            "Enroll": [721, 2367],
            "Top10perc": [23, 34],
            "F.Undergrad": [2885, 9919],
        }
    )
    analysis = _fallback_analysis("What information does this dataset contain?", list(df.columns))
    result, _, _ = execute_analysis(analysis.code, df)
    answer = _fallback_summary(analysis.plan, result, False)
    assert "colleges or universities" in answer
    assert "Unnamed: 0" not in answer
    assert "applications received" in answer
    assert "accepted applicants" in answer
    assert "percentage of new students from the top 10%" in answer
    assert "full-time undergraduates" in answer


def test_fallback_lists_some_colleges_from_unnamed_label_column() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": ["Abilene Christian University", "Baylor University"],
            "Private": ["Yes", "Yes"],
            "Apps": [1660, 6075],
        }
    )
    analysis = _fallback_analysis("list some colleges from the dataset", list(df.columns))
    assert analysis.classification == "Dataset Information"
    assert analysis.code is not None
    result, tables, _ = execute_analysis(analysis.code, df)
    assert result[0]["Unnamed: 0"] == "Abilene Christian University"
    assert tables[0].columns == ["Unnamed: 0"]


@pytest.mark.parametrize("question", ["Who won the NBA Finals?", "What is the weather?", "Explain quantum physics."])
def test_fallback_refuses_unrelated_questions(question: str) -> None:
    analysis = _fallback_analysis(question, COLLEGE_COLUMNS)
    assert analysis.classification == "Unsupported Request"
    assert analysis.plan.grounding_status == "Unsupported"
    assert analysis.code is None
