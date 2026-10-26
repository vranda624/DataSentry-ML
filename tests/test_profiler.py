import pandas as pd
from datasentry.profiler import profile_dataframe, build_quality_findings


def test_profile_counts_missing_and_duplicates():
    df = pd.DataFrame({"x": [1, 2, 2], "y": ["a", None, "a"]})
    result = profile_dataframe(df)
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 0


def test_findings_detect_missing_values():
    df = pd.DataFrame({"x": [1, None, 3, 4]})
    findings = build_quality_findings(df)
    assert "Missing values" in findings["issue"].tolist()
